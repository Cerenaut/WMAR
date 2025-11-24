
import socket
import torch.nn.utils as nn_utils
import types
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import pandas as pd
from config import Config
from generate_trajectory import (
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n_sac,
)
from tianshou.policy import DiscreteSACPolicy
from tianshou.data import Batch, ReplayBuffer
import time
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"gym\.utils\.passive_env_checker"
)
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"gym\.utils\.passive_env_checker"
)
import logging
# silence everything below ERROR in passive_env_checker
logging.getLogger("gym.utils.passive_env_checker").setLevel(logging.ERROR)


class CoinRunCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        output_dim,
        return_state=False,
        cnn_depth=32,
        mlp_features=512,
        mlp_layers=2,
    ):
        super().__init__()
        c, h, w = input_shape
        d = cnn_depth

        # Convolutional backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(c,     d, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(d,   2*d, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(2*d, 2*d, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Figure out flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        # MLP head before action logits
        layers = []
        in_dim = n_flatten
        for _ in range(mlp_layers):
            layers.append(nn.Linear(in_dim, mlp_features))
            layers.append(nn.ReLU())
            in_dim = mlp_features
        self.fc = nn.Sequential(*layers)

        # Final action logits layer
        self.head = nn.Linear(in_dim, output_dim)

        self.return_state = return_state

    def forward(self, x, state=None, info={}):
        # Convert numpy arrays to torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.parameters()).device)

        # Normalize if uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Add batch dim if missing
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Forward pass
        features = self.cnn(x)
        features = self.fc(features)
        out = self.head(features)

        if self.return_state:
            return out, state
        else:
            return out


def train_sac(config: Config):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"{current_time}_{socket.gethostname()}_seed{config.seed}"
    log_root = Path("runs") / "sac"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    config.save(log_dir / "config.json")
    writer = SummaryWriter(log_dir=str(log_dir))

    # environment
    schedule = config.get_env_schedule()

    # Tianshou replay buffer for off-policy updates
    ts_buffer = ReplayBuffer(size=config.sac_dv3_data_n_max, device=device)

    # Tianshou Discrete SAC setupa
    sample_env = schedule.funcs()[0]()
    
    # Gym gives HWC, PyTorch needs CHW.
    obs_shape = sample_env.observation_space.shape 
    if len(obs_shape) == 3 and obs_shape[0] in [64, 128]:  # (H, W, C)
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    action_n = sample_env.action_space.n

    actor_net   = CoinRunCNN(
        obs_shape, action_n, return_state=True,
        cnn_depth    = config.cnn_depth,
        mlp_features = config.mlp_features,
        mlp_layers   = config.mlp_layers,
    ).to(device)
    
    # For critics (just Q-value tensor)
    critic1_net = CoinRunCNN(
        obs_shape, action_n, return_state=False,
        cnn_depth    = config.cnn_depth,
        mlp_features = config.mlp_features,
        mlp_layers   = config.mlp_layers,
    ).to(device)
    critic2_net = CoinRunCNN(
        obs_shape, action_n, return_state=False,
        cnn_depth    = config.cnn_depth,
        mlp_features = config.mlp_features,
        mlp_layers   = config.mlp_layers,
    ).to(device)

    critic1_opt = Adam(critic1_net.parameters(), lr=config.sac_lr)
    critic2_opt = Adam(critic2_net.parameters(), lr=config.sac_lr)
    actor_opt   = Adam(actor_net.parameters(),   lr=config.sac_lr)

    # — Gradient-clipping wrapper for optimizers —
    def _clip_and_step(self, closure=None):
        torch.nn.utils.clip_grad_norm_(
            [p for g in self.param_groups for p in g['params']],
            max_norm=1.0
        )
        return self._orig_step(closure) if closure else self._orig_step()

    for opt in (actor_opt, critic1_opt, critic2_opt):
        opt._orig_step = opt.step
        opt.step       = types.MethodType(_clip_and_step, opt)
    # — end clipping wrapper —

    policy = DiscreteSACPolicy(
        actor_net,   actor_opt,
        critic1_net,     critic1_opt,
        critic2_net,     critic2_opt,
        tau=config.sac_tau,
        gamma=config.sac_gamma,
        alpha=config.sac_alpha,
    ).to(device)

    eval_history = []
    global_step = 0
    env_steps = 0
    best_rews_mean = -float("inf")
    #pbar = trange(config.epochs, desc="Epochs")

    for epoch in range(config.epochs):
        start = time.time()
        if config.random_policy == "first":
            random_policy = epoch == 0
        elif config.random_policy == "new":
            random_policy = schedule.is_new_env()

        # data collection
        for _ in range(
            config.pretrain_data_multiplier if (random_policy and config.pretrain_enabled) else 1
        ):
            
            acts, obss, rews, conts, resets = reinterpret_nt_to_t_n_sac(
                *generate_trajectories(
                    config.n_sync * config.gen_seq_len,
                    config.n_sync,
                    wm=None,
                    ac=None if random_policy else policy,
                    env_fns=schedule.funcs(),
                    env_repeat=config.env_repeat,
                ),
                config.data_t,
                config.data_n,
            )
            next_obss = torch.cat([obss[1:], obss[-1:]], dim=0)
            obs_np  = obss.cpu().numpy()
            act_np  = acts.cpu().numpy()
            rew_np  = rews.cpu().numpy()
            term_np = resets.cpu().numpy()
            next_np = next_obss.cpu().numpy()
            #print(f"[DEBUG] Pre‑flatten shapes → obs: {obs_np.shape}, act: {act_np.shape}, rew: {rew_np.shape}, term: {term_np.shape}, next_obs: {next_np.shape}")

            T, N = obs_np.shape[0], obs_np.shape[1]
            env_steps += T * N 
            B = T * N
            num_updates = B
            # reshape into B transitions
            obs_arr  = obs_np.reshape((B,) + obs_np.shape[2:])  
            #print(f"[DEBUG] obs_arr shape: {obs_arr.shape}")
 
            act_arr  = act_np.reshape((B,) + act_np.shape[2:])   
            if act_arr.ndim == 1:
                act_arr = act_arr[:, None]                        

            rew_arr  = rew_np.reshape(B)                         
            term_arr = term_np.reshape(B)
            cont_arr  = conts.cpu().numpy().reshape(B)            
            next_arr = next_np.reshape((B,) + next_np.shape[2:]) # [B, C, H, W]
            
            #print(f"[DEBUG] Flattened shapes → obs: {obs_arr.shape}, act: {act_arr.shape}, rew: {rew_arr.shape}, term: {term_arr.shape}, next_obs: {next_arr.shape}")

            rew_list   = rew_arr.tolist()    
            term_list  = term_arr.tolist()   

            trunc_list = [
                (cont_arr[i] == 0) and (not term_list[i])
                for i in range(B)
            ]


            for _, (o_i, a_i, r_i, d_i, t_i, o2_i) in enumerate(
                zip(obs_arr, act_arr, rew_list, term_list, trunc_list, next_arr)
            ):
                if isinstance(a_i, np.ndarray):
                    act_scalar = int(a_i.squeeze())
                else:
                    act_scalar = int(a_i)
                ts_buffer.add(Batch(
                    obs=        o_i,
                    act=        act_scalar,   # now shape [B]
                    rew=        r_i,
                    terminated=d_i,
                    truncated= t_i,
                    obs_next=   o2_i
                ))
            # debug prints 
            #print(f"[DEBUG]  terminals: {sum(term_list)}  truncations: {sum(trunc_list)}  total: {B}")
            #zeros = int((cont_arr == 0).sum())
            #print(f"[DEBUG] cont_arr zeros: {zeros}")
            #print(f"[DEBUG] term+trunc = {sum(term_list) + sum(trunc_list)}")
            #end_arr = ((cont_arr == 0) | (term_arr == 1))
            #print(f"[DEBUG]  total ends by cont|term: {int(end_arr.sum())}")
            #print(f"[DEBUG]  term+trunc         : {sum(term_list) + sum(trunc_list)}")


        schedule.step()

        writer.add_scalar("Sample/total_train_iters", global_step, global_step)
        writer.add_scalar("Sample/env_frames",       env_steps,   global_step)
        rews_eps_mean = rews.sum().item() / resets.sum().item()
        writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
        len_eps_mean = config.gen_seq_len / resets.sum().item() * config.env_repeat
        writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)

        # evaluation every 10 epochs
        if epoch % 1 == 0:
            eval_means, eval_stds = [], []
            for efns in schedule.eval_funcs():
                m, s = evaluate(
                    config.n_sync,
                    wm=None,
                    ac=policy,
                    env_fns=efns,
                    env_repeat=config.env_repeat,
                    n_rollouts=256,
                )
                eval_means.append(m)
                eval_stds.append(s)
            writer.add_scalars(
                "Perf/eval_rew_eps_mean", {f"{i}": v for i, v in enumerate(eval_means)}, global_step
            )
            writer.add_scalars(
                "Perf/eval_rew_eps_std",  {f"{i}": s for i, s in enumerate(eval_stds)}, global_step
            )
            
            for mean_i, std_i in zip(eval_means, eval_stds):
                eval_history.append({
                    "env_steps": env_steps,
                    "global_step": global_step,
                    "Perf/eval_rew_eps_mean": mean_i,
                    "Perf/eval_rew_eps_std":  std_i
                })
            df = pd.DataFrame(eval_history)
            tqdm.write(df.to_string())
            csv_path = f"{log_dir}/eval_history.csv"
            df.to_csv(csv_path, index=False)

    
            
            approx_perf = float(np.mean(eval_means))
            writer.add_scalar("Perf/approx_perf", approx_perf, epoch)
            if approx_perf >= best_rews_mean:
                best_rews_mean = approx_perf
                torch.save(actor_net.state_dict(),   log_dir / "actor_best.pth")
                torch.save(critic1_net.state_dict(), log_dir / "critic1_best.pth")
                torch.save(critic2_net.state_dict(), log_dir / "critic2_best.pth")
        schedule.step()
        # SAC updates
        for _ in range(num_updates):
            losses = policy.update(
                config.sac_batch_size,
                ts_buffer
            )

            if global_step % config.log_frequency == 0:
                writer.add_scalar("Metric/actor_loss",   losses['loss/actor'],   global_step)
                writer.add_scalar("Metric/critic1_loss", losses['loss/critic1'], global_step)
                writer.add_scalar("Metric/critic2_loss", losses['loss/critic2'], global_step)

                actor_loss   = losses['loss/actor'].item()   if hasattr(losses['loss/actor'], 'item')   else losses['loss/actor']
                critic1_loss = losses['loss/critic1'].item() if hasattr(losses['loss/critic1'], 'item') else losses['loss/critic1']
                critic2_loss = losses['loss/critic2'].item() if hasattr(losses['loss/critic2'], 'item') else losses['loss/critic2']

                print(
                    f"[env_steps {env_steps:04d}] "
                    f"[global_step {global_step:04d}] "
                    f"actor_loss={actor_loss:.4f}  "
                    f"critic1_loss={critic1_loss:.4f}  "
                    f"critic2_loss={critic2_loss:.4f}"
                )

            global_step += 1

        epoch_time = time.time() - start
        #pbar.set_postfix(epoch_time=f"{epoch_time:.2f}s")
        print(f"Epoch {epoch+1}/{config.epochs} finished in {epoch_time:.2f}s", flush=True)



    # FINAL CHECKPOINT
    torch.save(actor_net.state_dict(),   log_dir / "actor.pth")
    torch.save(critic1_net.state_dict(), log_dir / "critic1.pth")
    torch.save(critic2_net.state_dict(), log_dir / "critic2.pth")
    writer.close()


"""

import socket
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config import Config
from generate_trajectory import evaluate
from tianshou.policy import DiscreteSACPolicy
from gymnasium.vector import SyncVectorEnv
from tianshou.data import Collector, ReplayBuffer
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import BasicLogger


class CoinRunCNN(nn.Module):
    def __init__(
        self,
        input_shape,
        output_dim,
        return_state=False,
        cnn_depth=32,
        mlp_features=512,
        mlp_layers=2,
    ):
        super().__init__()
        c, h, w = input_shape
        d = cnn_depth

        # Convolutional backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(c,     d, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(d,   2*d, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(2*d, 2*d, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )

        # Figure out flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]

        # MLP head before action logits
        layers = []
        in_dim = n_flatten
        for _ in range(mlp_layers):
            layers.append(nn.Linear(in_dim, mlp_features))
            layers.append(nn.ReLU())
            in_dim = mlp_features
        self.fc = nn.Sequential(*layers)

        # Final action logits layer
        self.head = nn.Linear(in_dim, output_dim)

        self.return_state = return_state

    def forward(self, x, state=None, info={}):
        # Convert numpy arrays to torch
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.parameters()).device)

        # Normalize if uint8
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # Add batch dim if missing
        if x.ndim == 3:
            x = x.unsqueeze(0)

        # Forward pass
        features = self.cnn(x)
        features = self.fc(features)
        out = self.head(features)

        if self.return_state:
            return out, state
        else:
            return out


def train_sac(config: Config):
    # 1) device & seeds
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 2) logging setup
    now = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"{now}_{socket.gethostname()}_seed{config.seed}"
    log_root = Path("runs") / "sac"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    config.save(log_dir / "config.json")
    writer = SummaryWriter(str(log_dir))

    # 3) build schedule and sample env for shapes
    schedule = config.get_env_schedule()
    sample_env = schedule.funcs()[0]()
    obs_shape = sample_env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[0] in [64, 128]:
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
    action_n = sample_env.action_space.n

    # 4) build networks
    actor_net = CoinRunCNN(
        obs_shape, action_n, return_state=True,
        cnn_depth=config.cnn_depth,
        mlp_features=config.mlp_features,
        mlp_layers=config.mlp_layers,
    ).to(device)
    critic1_net = CoinRunCNN(
        obs_shape, action_n, return_state=False,
        cnn_depth=config.cnn_depth,
        mlp_features=config.mlp_features,
        mlp_layers=config.mlp_layers,
    ).to(device)
    critic2_net = CoinRunCNN(
        obs_shape, action_n, return_state=False,
        cnn_depth=config.cnn_depth,
        mlp_features=config.mlp_features,
        mlp_layers=config.mlp_layers,
    ).to(device)

    # 5) optimizers & auto‐α
    critic1_opt = Adam(critic1_net.parameters(), lr=1e-4)
    critic2_opt = Adam(critic2_net.parameters(), lr=1e-4)
    actor_opt = Adam(actor_net.parameters(), lr=1e-4)
    target_entropy = np.log(action_n)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optim = Adam([log_alpha], lr=3e-4)

    # 6) policy with auto‐tuned α & reward norm
    policy = DiscreteSACPolicy(
        actor_net, actor_opt,
        critic1_net, critic1_opt,
        critic2_net, critic2_opt,
        tau=config.sac_tau,
        gamma=config.sac_gamma,
        alpha=(target_entropy, log_alpha, alpha_optim),
        reward_normalization=True,
    ).to(device)

    base_env_fn = schedule.funcs()[0]

    sample_env  = base_env_fn()
    from gymnasium.vector import SyncVectorEnv

    # 7) vectorized envs
    train_envs = SyncVectorEnv([base_env_fn for _ in range(config.n_sync)])
    test_envs  = SyncVectorEnv([base_env_fn for _ in range(config.n_sync)])
    # 7) vectorized envs – wrap to inject a dummy `render_mode`
    def make_env():
        env = base_env_fn()
        # Gymnasium’s SyncVectorEnv will immediately do `env.render_mode`
        # so define it here (even if None)
        if not hasattr(env, "render_mode"):
            env.render_mode = None
        return env

    train_envs = SyncVectorEnv([make_env for _ in range(config.n_sync)])
    test_envs  = SyncVectorEnv([make_env for _ in range(config.n_sync)])

    # 8) buffer & collectors
    buffer = ReplayBuffer(size=config.sac_dv3_data_n_max, device=device)
    train_collector = Collector(
        policy, train_envs, buffer,
        exploration_noise=True,
        reward_normalization=True,
    )
    test_collector = Collector(
        policy, test_envs,
        exploration_noise=False,
        reward_normalization=True,
    )

    # 9) prepare evaluation history
    eval_history = []

    # 10) define custom test_fn
    def test_fn(epoch: int, env_step: int, gradient_step: int):
        eval_means, eval_stds = [], []
        for efns in schedule.eval_funcs():
            m, s = evaluate(
                config.n_sync,
                wm=None,
                ac=policy,
                env_fns=efns,
                env_repeat=config.env_repeat,
                n_rollouts=256,
            )
            eval_means.append(m)
            eval_stds.append(s)

        # TensorBoard logging
        writer.add_scalars(
            "Perf/eval_rew_eps_mean",
            {f"{i}": v for i, v in enumerate(eval_means)},
            env_step
        )
        writer.add_scalars(
            "Perf/eval_rew_eps_std",
            {f"{i}": s for i, s in enumerate(eval_stds)},
            env_step
        )

        # append CSV entries
        for i, (mean_i, std_i) in enumerate(zip(eval_means, eval_stds)):
            eval_history.append({
                "env_steps": env_step,
                "global_step": gradient_step,
                "env_index": i,
                "Perf/eval_rew_eps_mean": mean_i,
                "Perf/eval_rew_eps_std": std_i
            })

        # dump CSV and print
        df = pd.DataFrame(eval_history)
        df.to_csv(log_dir / "eval_history.csv", index=False)
        print(df.to_string())

        # return scalar for best-model tracking
        return float(np.mean(eval_means))

    # 11) run training
    result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        max_epoch=config.epochs,
        step_per_epoch=config.steps_per_batch,
        repeat_per_collect=config.ac_train_steps,
        episode_per_test=config.n_sync,
        batch_size=config.sac_batch_size,
        logger_kwargs={'logger': BasicLogger(writer)},
        train_fn=lambda e, s: schedule.step(),
        test_fn=test_fn,
        test_in_train=False,
    )

    # 12) save final models
    torch.save(actor_net.state_dict(), log_dir / "actor.pth")
    torch.save(critic1_net.state_dict(), log_dir / "critic1.pth")
    torch.save(critic2_net.state_dict(), log_dir / "critic2.pth")
    writer.close()
    return result
"""