
import argparse
import json
import os
import socket
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange
import replay
from config import Config
from generate_trajectory import (
    SequentialEnvironments,
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)

# NETWORK DEFINITIONS (WMAR encoder + MLP widths)
# ---------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, config: Config):
        super().__init__()
        from vae import Encoder
        self.encoder = Encoder(img_channels=in_channels, channels=config.cnn_depth)
        enc_out = self.encoder.output_size
        hidden = config.mlp_features
        layers = []
        for i in range(config.mlp_layers):
            if i == 0:
                layers.append(nn.Linear(enc_out + action_dim, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs / 255.0)
        x = torch.cat([z, action], dim=-1)
        return self.fc(x)


class GaussianPolicy(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, config: Config):
        super().__init__()
        from vae import Encoder
        self.encoder = Encoder(img_channels=in_channels, channels=config.cnn_depth)
        enc_out = self.encoder.output_size
        hidden = config.mlp_features
        layers = []
        for _ in range(config.mlp_layers):
            layers.append(nn.Linear(enc_out, hidden))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor):
        z = self.encoder(obs / 255.0)
        h = self.mlp(z)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(-20, 2)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, obs: torch.Tensor):
        mean, std = self(obs)
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


# SAC TRAINING LOOP
# ---------------------------------------------------------
def train_sac(config: Config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # build run_name & log_dir
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    run_name = f"{current_time}_{socket.gethostname()}_seed{config.seed}"
    log_root = Path("runs") / "sac"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config.save(log_dir / "config.json")

    writer = SummaryWriter(log_dir=str(log_dir))

    # environment schedule & replay buffer
    schedule = config.get_env_schedule()
    replay_buffer = config.get_replay_buffer()

    # networks & targets
    dummy = torch.zeros(1, 3, config.img_size, config.img_size).cuda()
    in_ch = dummy.shape[1]
    act_dim = config.action_space
    policy = GaussianPolicy(in_ch, act_dim, config).cuda()
    q1 = QNetwork(in_ch, act_dim, config).cuda()
    q2 = QNetwork(in_ch, act_dim, config).cuda()
    target_q1 = QNetwork(in_ch, act_dim, config).cuda()
    target_q2 = QNetwork(in_ch, act_dim, config).cuda()
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    policy_opt = Adam(policy.parameters(), lr=config.sac_lr)
    q1_opt = Adam(q1.parameters(), lr=config.sac_lr)
    q2_opt = Adam(q2.parameters(), lr=config.sac_lr)

    total_env_steps = 0
    global_step = 0
    best_rews_mean = -float("inf")
    
    Q1_loss_val, Q2_loss_val, policy_loss_val = None, None, None
    # main loop
    for epoch in range(config.epochs):
        
        # 1) DATA COLLECTION
        acts, obss, rews, conts, resets = reinterpret_nt_to_t_n(
            *generate_trajectories(
                config.n_sync * config.gen_seq_len,
                config.n_sync,
                wm=None,
                ac=policy,
                env_fns=schedule.funcs(),
                env_repeat=config.env_repeat,
            ),
            config.data_t,
            config.data_n,
        )
        replay_buffer.add(acts, obss, rews, conts, resets)
        schedule.step()

        #  metrics
        num_new_env_steps = acts.shape[0] * acts.shape[1] * config.env_repeat
        total_env_steps += num_new_env_steps
        writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)
        writer.add_scalar("Sample/replay_buffer_size", replay_buffer.n_valid, global_step)
        writer.add_scalar("Sample/total_train_iters", global_step, global_step)

        rews_eps_mean = rews.sum().item() / resets.sum().item()
        writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
        len_eps_mean = config.gen_seq_len / resets.sum().item() * config.env_repeat
        writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)
        
        Q1_loss_val = None
        Q2_loss_val = None
        policy_loss_val = None

        # SAC UPDATES
        for _ in range(config.ac_train_steps):
            if replay_buffer.n_valid < config.sac_batch_size:
                break
            mb_acts, mb_obs, mb_rews, mb_conts, mb_resets = replay_buffer.minibatch(
                mb_t=1, mb_n=config.sac_batch_size, mb_device="cuda"
            )
            mb_obs = mb_obs.squeeze(0)
            mb_acts = mb_acts.squeeze(0)
            mb_rews = mb_rews.squeeze(0)
            nonterm = (1 - mb_resets.squeeze(0)).float().cuda()

            # targets
            with torch.no_grad():
                next_a, next_logp = policy.sample(mb_obs)
                tq1 = target_q1(mb_obs, next_a)
                tq2 = target_q2(mb_obs, next_a)
                target_Q = mb_rews + config.sac_gamma * nonterm * (
                    torch.min(tq1, tq2) - config.sac_alpha * next_logp
                )
            # Q losses
            q1_loss = nn.MSELoss()(q1(mb_obs, mb_acts), target_Q)
            q2_loss = nn.MSELoss()(q2(mb_obs, mb_acts), target_Q)
            q1_opt.zero_grad(); q1_loss.backward(); q1_opt.step()
            q2_opt.zero_grad(); q2_loss.backward(); q2_opt.step()
            
            Q1_loss_val     = q1_loss.item()
            Q2_loss_val     = q2_loss.item()
            policy_loss_val = policy_loss.item()
            
            # policy loss
            new_a, logp_pi = policy.sample(mb_obs)
            q_pi = torch.min(q1(mb_obs, new_a), q2(mb_obs, new_a))
            policy_loss = (config.sac_alpha * logp_pi - q_pi).mean()
            policy_opt.zero_grad(); policy_loss.backward(); policy_opt.step()
            # soft updates
            for p, tp in zip(q1.parameters(), target_q1.parameters()):
                tp.data.mul_(1 - config.sac_tau); tp.data.add_(config.sac_tau * p.data)
            for p, tp in zip(q2.parameters(), target_q2.parameters()):
                tp.data.mul_(1 - config.sac_tau); tp.data.add_(config.sac_tau * p.data)

        # Metric logs
        grad_norm = torch.nn.utils.clip_grad_norm_(q1.parameters(), 1000)
        writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
        if Q1_loss_val is not None:
            writer.add_scalar("Metric/Q1_loss",    Q1_loss_val,    global_step)
            writer.add_scalar("Metric/Q2_loss",    Q2_loss_val,    global_step)
            writer.add_scalar("Metric/Policy_loss", policy_loss_val, global_step)

        # EVALUATION every 10 epochs
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
                "Perf/eval_rew_eps_mean",
                {f"{i}": v for i, v in enumerate(eval_means)},
                global_step,
            )
            writer.add_scalars(
                "Perf/eval_rew_eps_std",
                {f"{i}": v for i, v in enumerate(eval_stds)},
                global_step,
            )
            approx_perf = float(np.mean(eval_means))
            writer.add_scalar("Perf/approx_perf", approx_perf, global_step)
            if approx_perf >= best_rews_mean:
                best_rews_mean = approx_perf
                torch.save(policy.state_dict(), log_dir / "save_sac_policy_best.pt")
                torch.save(q1.state_dict(), log_dir / "save_sac_q1_best.pt")
                torch.save(q2.state_dict(), log_dir / "save_sac_q2_best.pt")

        print(f"Finished epoch {epoch+1}/{config.epochs}")


        global_step += 1

    # final save
    torch.save(policy.state_dict(), log_dir / "policy.pt")
    torch.save(q1.state_dict(), log_dir / "q1.pt")
    torch.save(q2.state_dict(), log_dir / "q2.pt")
    writer.close()



