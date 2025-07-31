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

import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path
from vae import Encoder
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
import pandas as pd
import replay
from config import Config
from generate_trajectory import (
    SequentialEnvironments,
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)
from tianshou.policy import DiscreteSACPolicy
from tianshou.data import Batch, ReplayBuffer


class CoinRunCNN(nn.Module):
    def __init__(self, input_shape, output_dim, return_state=False):
        super().__init__()
        c, h, w = input_shape
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(dummy).shape[1]
        self.head = nn.Linear(n_flatten, output_dim)
        self.return_state = return_state

    def forward(self, x, state=None, info={}):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(next(self.parameters()).device)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        if x.ndim == 3:
            x = x.unsqueeze(0)
        out = self.head(self.cnn(x))
        if self.return_state:
            return out, state
        else:
            return out


def train_sac(config: Config):
    # detect device (MPS > CUDA > CPU), mps for Mac m1,m2, ...
    if torch.backends.mps.is_available():
        device = torch.device('cpu')
    elif torch.cuda.is_available():
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

    # Tianshou Discrete SAC setup
    sample_env = schedule.funcs()[0]()
    
    # Gym gives HWC, PyTorch needs CHW.
    obs_shape = sample_env.observation_space.shape 
    if len(obs_shape) == 3 and obs_shape[0] in [64, 128]:  # (H, W, C)
        obs_shape = (obs_shape[2], obs_shape[0], obs_shape[1])  # (C, H, W)
    action_n = sample_env.action_space.n

    actor_net   = CoinRunCNN(obs_shape, action_n, return_state=True).to(device)
    
    #  sanity‐check network I/O 
    with torch.no_grad():
        sample = torch.randn(32, *obs_shape).to(device)
        logits, _ = actor_net(sample)
        print(f"[SANITY] actor_net output shape: {logits.shape}") 



    # For critics (just Q-value tensor)
    critic1_net = CoinRunCNN(obs_shape, action_n, return_state=False).to(device)
    critic2_net = CoinRunCNN(obs_shape, action_n, return_state=False).to(device)

    with torch.no_grad():
        q1 = critic1_net(sample)
        print(f"[SANITY] critic1_net output shape: {q1.shape}") 

    critic1 = critic1_net
    critic2 = critic2_net

    critic1_opt = Adam(critic1.parameters(), lr=config.sac_lr)
    critic2_opt = Adam(critic2.parameters(), lr=config.sac_lr)
    actor_opt   = Adam(actor_net.parameters(),   lr=config.sac_lr)

    policy = DiscreteSACPolicy(
        actor_net,   actor_opt,
        critic1,     critic1_opt,
        critic2,     critic2_opt,
        tau=config.sac_tau,
        gamma=config.sac_gamma,
        alpha=config.sac_alpha,
    ).to(device)


    eval_history = []
    global_step = 0
    best_rews_mean = -float("inf")
    for epoch in trange(config.epochs, desc="Epochs"):

        if config.random_policy == "first":
            random_policy = epoch == 0
        elif config.random_policy == "new":
            random_policy = schedule.is_new_env()

        # data collection
        for _ in range(
            config.pretrain_data_multiplier if (random_policy and config.pretrain_enabled) else 1
        ):
            
            acts, obss, rews, conts, resets = reinterpret_nt_to_t_n(
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
            B = T * N

            # reshape into B transitions
            obs_arr  = obs_np.reshape((B,) + obs_np.shape[2:])  
            print(f"[DEBUG] obs_arr shape: {obs_arr.shape}")
 
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
            print(f"[DEBUG]  terminals: {sum(term_list)}  truncations: {sum(trunc_list)}  total: {B}")
            zeros = int((cont_arr == 0).sum())
            print(f"[DEBUG] cont_arr zeros: {zeros}")
            print(f"[DEBUG] term+trunc = {sum(term_list) + sum(trunc_list)}")
            end_arr = ((cont_arr == 0) | (term_arr == 1))
            print(f"[DEBUG]  total ends by cont|term: {int(end_arr.sum())}")
            print(f"[DEBUG]  term+trunc         : {sum(term_list) + sum(trunc_list)}")


        schedule.step()

        writer.add_scalar("Sample/total_train_iters", global_step, global_step)

        rews_eps_mean = rews.sum().item() / resets.sum().item()
        writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
        len_eps_mean = config.gen_seq_len / resets.sum().item() * config.env_repeat
        writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)

        # evaluation every 10 epochs
        if epoch % 10 == 0:
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
                    "step": global_step,
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
                torch.save(critic1.state_dict(), log_dir / "critic1_best.pth")
                torch.save(critic2.state_dict(), log_dir / "critic2_best.pth")
        schedule.step()
        # SAC updates
        for _ in range(config.steps_per_batch): # config.steps_per_batch = 1000 in configs files which is too low for sac
            losses = policy.update(
                config.sac_batch_size,
                ts_buffer
            )

            if global_step % config.log_frequency == 0:
                writer.add_scalar("Metric/actor_loss",   losses['loss/actor'],   global_step)
                writer.add_scalar("Metric/critic1_loss", losses['loss/critic1'], global_step)
                writer.add_scalar("Metric/critic2_loss", losses['loss/critic2'], global_step)
            global_step += 1


    # FINAL CHECKPOINT
    torch.save(actor_net.state_dict(),   log_dir / "actor.pth")
    torch.save(critic1.state_dict(), log_dir / "critic1.pth")
    torch.save(critic2.state_dict(), log_dir / "critic2.pth")
    writer.close()
