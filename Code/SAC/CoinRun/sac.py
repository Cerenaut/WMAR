# sac.py
import os
import socket
from datetime import datetime
from pathlib import Path
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from config import Config
from generate_trajectory import (
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)
from vae import Encoder

from typing import Optional


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, config: Config, encoder: Optional[Encoder] = None):
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder(img_channels=in_channels, channels=config.cnn_depth)
        enc_out = self.encoder.output_size
        hidden = config.mlp_features
        layers = []
        for i in range(config.mlp_layers):
            if i == 0:
                layers.append(nn.Linear(enc_out, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden, action_dim))
        self.fc = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Inputs are already normalized to [0,1]
        z = self.encoder(obs)
        q_values = self.fc(z)
        return q_values  # shape: (B, action_dim)


class CategoricalPolicy(nn.Module):
    def __init__(self, in_channels: int, action_dim: int, config: Config, encoder: Optional[Encoder] = None):
        super().__init__()
        
        self.encoder = encoder if encoder is not None else Encoder(img_channels=in_channels, channels=config.cnn_depth)
        enc_out = self.encoder.output_size
        hidden = config.mlp_features
        layers = []
        for i in range(config.mlp_layers):
            if i == 0:
                layers.append(nn.Linear(enc_out, hidden))
            else:
                layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*layers)
        self.logits_layer = nn.Linear(hidden, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        h = self.mlp(z)
        logits = self.logits_layer(h)
        return logits  # shape: (B, action_dim)

    def probs_and_logprobs(self, obs: torch.Tensor):
        logits = self(obs)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        return probs, log_probs

    def sample_action_indices(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample()  # shape: (B,)

def train_sac(config: Config):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build run_name & log_dir
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    job_id = os.getenv("SLURM_JOB_ID")
    run_name = f"{current_time}_{socket.gethostname()}_{config.seed}_{job_id}"
    print(f"train_sac: run_name={run_name}")
    # Use absolute log directory to avoid issues on HPC with changing CWD or cleanup
    log_root = Path.cwd() / "runs" / "reveresed_order"
    log_root.mkdir(parents=True, exist_ok=True)
    log_dir = log_root / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"train_sac: log_dir={log_dir}")
    config.save(log_dir / "config.json")
    writer = SummaryWriter(log_dir=str(log_dir.resolve()), flush_secs=10)



    schedule = config.get_env_schedule()
    replay_buffer = config.get_replay_buffer()


    # use 32×32 resolution as per ARROW
    dummy = torch.zeros(1, 3, 32, 32, device=device)
    in_ch = dummy.shape[1]
    act_dim = config.action_space
    
    critic_shared_encoder = Encoder(img_channels=in_ch, channels=config.cnn_depth).to(device)
    # Actor gets its own encoder to avoid gradient interference
    policy = CategoricalPolicy(in_ch, act_dim, config, encoder=None).to(device)
    q1 = QNetwork(in_ch, act_dim, config, encoder=critic_shared_encoder).to(device)
    q2 = QNetwork(in_ch, act_dim, config, encoder=critic_shared_encoder).to(device)
    target_q1 = QNetwork(in_ch, act_dim, config).to(device)
    target_q2 = QNetwork(in_ch, act_dim, config).to(device)
    
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    # Same LR for actor and critics by default; allow faster actor via sac_policy_lr
    policy_lr = getattr(config, "sac_policy_lr", config.sac_lr)
    policy_opt = Adam(policy.parameters(), lr=policy_lr)
    q1_opt = Adam(q1.parameters(), lr=config.sac_lr)
    q2_opt = Adam(q2.parameters(), lr=config.sac_lr)
    # Entropy coefficient (alpha) autotuning
    # Discrete SAC: target entropy should be positive (≈ log(|A|))
    target_entropy = config.sac_target_entropy_coef * np.log(act_dim)
    log_alpha = torch.tensor(np.log(config.sac_alpha), device=device, requires_grad=True)
    alpha_opt = Adam([log_alpha], lr=config.sac_alpha_lr)

    total_env_steps = 0
    global_step = 0
    best_rews_mean = -float("inf")
    # TES-SAC state
    tes_enabled = getattr(config, "sac_tes_enabled", True)
    tes_lambda = getattr(config, "sac_tes_lambda", 0.999)
    tes_avg_th = getattr(config, "sac_tes_avg_threshold", 0.01)
    tes_std_th = getattr(config, "sac_tes_std_threshold", 0.05)
    tes_k = getattr(config, "sac_tes_discount_k", 0.9)
    tes_T = getattr(config, "sac_tes_T", 1000)
    tes_i = 0
    tes_start_epoch = getattr(config, "sac_tes_start_epoch", 0)
    ema_entropy = target_entropy  # initialize EMA around current target
    ema_var = 0.0
    print(f"TES: enabled={tes_enabled} lambda={tes_lambda} avg_th={tes_avg_th} std_th={tes_std_th} k={tes_k} T={tes_T}")
    print(f"train_sac: Training variables initialized")
    print(f"train_sac: total_env_steps={total_env_steps}, global_step={global_step}, best_rews_mean={best_rews_mean}")
    print(f"train_sac: Training config - epochs={config.epochs}, steps_per_batch={config.steps_per_batch}")
    print(f"train_sac: SAC config - gamma={config.sac_gamma}, alpha={config.sac_alpha}, tau={config.sac_tau}, batch_size={config.sac_batch_size}")

    
    
    for epoch in range(config.epochs):
        print(f"\n===== EPOCH {epoch}/{config.epochs} STARTING =====\n")

        
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
        print(f"train_sac: Rewards - rews.shape={rews.shape}, sum={rews.sum().item():.3f}, mean={rews.mean().item():.3f}")
        print(f"train_sac: Resets - resets.shape={resets.shape}, sum={resets.sum().item()}, episodes={resets.sum().item()}")
        
        replay_buffer.add(acts, obss, rews, conts, resets)

        # SAMPLE METRICS
        env_steps_this_epoch = acts.shape[0] * acts.shape[1] * config.env_repeat
        total_env_steps += env_steps_this_epoch
        print(f"train_sac: Env steps this epoch: {env_steps_this_epoch}, total: {total_env_steps}")
        try:
            writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)
            writer.add_scalar("Sample/replay_buffer_size", replay_buffer.n_valid, global_step)
            writer.add_scalar("Sample/total_train_iters", global_step, global_step)
        except Exception as tb_err:
            print(f"train_sac: TensorBoard write failed: {tb_err}")

        # PERFORMANCE METRICS
        rews_eps_mean = rews.sum().item() / resets.sum().item()
        len_eps_mean = config.gen_seq_len / resets.sum().item() * config.env_repeat
        try:
            print(f"train_sac: Performance metrics - rews_eps_mean={rews_eps_mean:.3f}, len_eps_mean={len_eps_mean:.3f}")
            writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
            writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)
        except Exception as tb_err:
            print(f"train_sac: TensorBoard write failed: {tb_err}")

        Q1_loss_val = Q2_loss_val = policy_loss_val = None

        # EVALUATION
        if epoch % 10 == 0:
            print(f"train_sac: Starting evaluation at epoch {epoch}")
            print("Evaluation started...")
            eval_means, eval_stds = [], []
            eval_funcs = schedule.eval_funcs()
            print(f"train_sac: Number of evaluation environments: {len(eval_funcs)}")
            
            for i, efns in enumerate(eval_funcs):
                print(f"train_sac: Evaluating environment {i+1}/{len(eval_funcs)}")
                m_pol, s_pol = evaluate(
                    config.n_sync,
                    wm=None,
                    ac=policy,
                    env_fns=efns,
                    env_repeat=config.env_repeat,
                    n_rollouts=256,
                )
                print(f"train_sac: Env {i+1} eval - (mean={m_pol:.3f}, std={s_pol:.3f})")
                eval_means.append(m_pol)
                eval_stds.append(s_pol)
                
            try:
                writer.add_scalars("Perf/eval_rew_eps_mean", {f"{i}": v for i, v in enumerate(eval_means)}, global_step)
                writer.add_scalars("Perf/eval_rew_eps_std", {f"{i}": s for i, s in enumerate(eval_stds)}, global_step)
            except Exception as tb_err:
                print(f"train_sac: TensorBoard write failed: {tb_err}")
            print(f"Eval means: {eval_means}")
            print(f"Eval stds: {eval_stds}")
            print(f"global_step: {global_step}")

            approx_perf = float(np.mean(eval_means))
            print(f"train_sac: Overall performance: {approx_perf:.3f} (previous best: {best_rews_mean:.3f})")
            try:
                writer.add_scalar("Perf/approx_perf", approx_perf, global_step)
            except Exception as tb_err:
                print(f"train_sac: TensorBoard write failed: {tb_err}")
            if approx_perf >= best_rews_mean:
                print(f"train_sac: New best performance! Saving models...")
                best_rews_mean = approx_perf
                torch.save(policy.state_dict(), log_dir / "save_sac_policy_best.pt")
                torch.save(q1.state_dict(),      log_dir / "save_sac_q1_best.pt")
                torch.save(q2.state_dict(),      log_dir / "save_sac_q2_best.pt")
                print(f"train_sac: Best models saved to {log_dir}")
            else:
                print(f"train_sac: Performance not improved, keeping previous best")
        
        

        # SAC UPDATES
        print(f"train_sac: Starting SAC updates for epoch {epoch}")


        q1_loss_sum = 0.0
        q2_loss_sum = 0.0
        policy_loss_sum = 0.0
        entropy_sum = 0.0
        last_q1_loss = None
        last_q2_loss = None
        last_policy_loss = None
        last_entropy = None
        for step in range(config.steps_per_batch):
            # two-step windows to build (s, a, r, s_next, nonterm)
            mb_acts, mb_obs, mb_rews, _, mb_resets = replay_buffer.minibatch(
                mb_t=2, mb_n=config.sac_batch_size, mb_device=device.type
            )
            s      = mb_obs[0].to(device)
            s_next = mb_obs[1].to(device)
            a_t    = mb_acts[1].to(device)
            r_t    = mb_rews[1].to(device)
            nonterm = (1 - mb_resets[1]).float().to(device)

            # targets
            with torch.no_grad():
                # Discrete SAC target on s_next
                next_logits = policy(s_next)
                next_log_probs = torch.log_softmax(next_logits, dim=-1)
                next_probs = torch.softmax(next_logits, dim=-1)
                tq1_all = target_q1(s_next)
                tq2_all = target_q2(s_next)
                min_q_next = torch.min(tq1_all, tq2_all)
                alpha = log_alpha.exp().detach()
                v_next = (next_probs * (min_q_next - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
                target_Q = r_t + config.sac_gamma * nonterm * v_next

            # Q losses and update
            # Compute Q losses
            # Q(s,a) gather on chosen actions (one-hot -> indices)
            act_idx = a_t.argmax(dim=-1) if a_t.dim() == 2 else a_t
            q1_all = q1(s)
            q2_all = q2(s)
            q1_pred = q1_all.gather(1, act_idx.long().unsqueeze(-1))
            q2_pred = q2_all.gather(1, act_idx.long().unsqueeze(-1))
            
            q1_loss = nn.MSELoss()(q1_pred, target_Q)
            q2_loss = nn.MSELoss()(q2_pred, target_Q)
            total_q_loss = q1_loss + q2_loss
            # Joint backward for shared encoder
            q1_opt.zero_grad(); q2_opt.zero_grad()
            total_q_loss.backward()
            torch.nn.utils.clip_grad_norm_(q1.parameters(), config.sac_grad_clip)
            torch.nn.utils.clip_grad_norm_(q2.parameters(), config.sac_grad_clip)
            q1_opt.step(); q2_opt.step()
            # Accumulate losses
            q1_loss_sum += float(q1_loss.item())
            q2_loss_sum += float(q2_loss.item())
            last_q1_loss = float(q1_loss.item())
            last_q2_loss = float(q2_loss.item())

            # policy loss and update
            # Policy loss: maximize E[Q - alpha*logpi] -> minimize alpha*logpi - Q
            logits = policy(s)
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            q1_all = q1(s)
            q2_all = q2(s)
            # Detach Q to avoid backpropagating policy gradients into Q nets via shared encoder
            q_min = torch.min(q1_all, q2_all).detach()
            alpha = log_alpha.exp().detach()
            policy_loss = (probs * (alpha * log_probs - q_min)).sum(dim=-1).mean()
            
            policy_opt.zero_grad(); policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.sac_grad_clip)
            policy_opt.step()
            
            # Alpha (entropy) loss update
            with torch.no_grad():
                entropy = -(probs * log_probs).sum(dim=-1).mean()
            # Correct sign for gradient descent: if entropy > target → decrease alpha; if entropy < target → increase alpha
            alpha_loss = log_alpha * (entropy - target_entropy)
            alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
            # Clamp log_alpha to avoid runaway in early training
            with torch.no_grad():
                log_alpha.clamp_(min=np.log(getattr(config, "sac_alpha_min", 1e-6)), max=np.log(getattr(config, "sac_alpha_max", 1.0)))
                # TES-SAC: update target_entropy by annealing when EMA entropy stabilizes
                if tes_enabled and (epoch >= tes_start_epoch):
                    # Exponential moving average and variance of policy entropy
                    delta = float(entropy.item()) - float(ema_entropy)
                    ema_entropy = float(ema_entropy) + (1.0 - tes_lambda) * delta
                    ema_var = tes_lambda * (ema_var + (1.0 - tes_lambda) * (delta ** 2))
                    ema_std = (ema_var ** 0.5)
                    tes_i += 1
                    # Check stability window and thresholds
                    if (abs(float(target_entropy) - ema_entropy) <= tes_avg_th) and (ema_std <= tes_std_th):
                        if tes_i >= tes_T:
                            old_te = float(target_entropy)
                            target_entropy = float(target_entropy) * tes_k
                            tes_i = 0
                            print(f"TES: anneal target_entropy {old_te:.4f} -> {float(target_entropy):.4f} (ema={ema_entropy:.4f}, std={ema_std:.4f})")
                    else:
                        # reset counter if not stable
                        tes_i = 0
            # Accumulate entropy metrics
            entropy_sum += float(entropy.item())
            last_entropy = float(entropy.item())
            # Accumulate losses
            policy_loss_sum += float(policy_loss.item())
            last_policy_loss = float(policy_loss.item())

            # soft updates
            # Soft target updates (no per-step debug)
            
            # Track parameter changes for debugging
            q1_param_change = 0.0
            q2_param_change = 0.0
            param_count = 0
            
            for p, tp in zip(q1.parameters(), target_q1.parameters()):
                old_tp = tp.data.clone()
                tp.data.mul_(1 - config.sac_tau)
                tp.data.add_(config.sac_tau * p.data)
                if step == 0:
                    q1_param_change += (tp.data - old_tp).abs().sum().item()
                    param_count += tp.numel()
                    
            for p, tp in zip(q2.parameters(), target_q2.parameters()):
                old_tp = tp.data.clone()
                tp.data.mul_(1 - config.sac_tau)
                tp.data.add_(config.sac_tau * p.data)
                if step == 0:
                    q2_param_change += (tp.data - old_tp).abs().sum().item()
            
            # Skip per-step parameter change reporting

            # Cache last-step losses
            Q1_loss_val     = last_q1_loss
            Q2_loss_val     = last_q2_loss
            policy_loss_val = last_policy_loss
            
            global_step += 1
            
        q1_loss_mean = q1_loss_sum / config.steps_per_batch
        q2_loss_mean = q2_loss_sum / config.steps_per_batch
        policy_loss_mean = policy_loss_sum / config.steps_per_batch
        entropy_mean = entropy_sum / config.steps_per_batch if last_entropy is not None else float('nan')
        alpha_current = float(torch.exp(log_alpha).item())
        print(
            f"train_sac: Updates summary - steps={config.steps_per_batch}, "
            f"Q1_mean={q1_loss_mean:.6f}, Q2_mean={q2_loss_mean:.6f}, Policy_mean={policy_loss_mean:.6f}, "
            f"alpha={alpha_current:.6f}, policy_entropy_mean={entropy_mean:.6f}"
        )
        # Use means as epoch losses for logging
        Q1_loss_val = q1_loss_mean
        Q2_loss_val = q2_loss_mean
        policy_loss_val = policy_loss_mean

        # METRIC LOGS
        # Compute gradient norm without mutating stored grads
        total_sq_norm = 0.0
        for p in q1.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                total_sq_norm += float(g.pow(2).sum().item())
        grad_norm = total_sq_norm ** 0.5
        print(f"train_sac: Gradient norm (Q1): {grad_norm:.6f}")
        try:
            writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
        except Exception as tb_err:
            print(f"train_sac: TensorBoard write failed: {tb_err}")
        
        if Q1_loss_val is not None:
            print(f"train_sac: Final epoch losses - Q1:{Q1_loss_val:.6f}, Q2:{Q2_loss_val:.6f}, Policy:{policy_loss_val:.6f}")
            try:
                writer.add_scalar("Metric/Q1_loss",    Q1_loss_val,    global_step)
                writer.add_scalar("Metric/Q2_loss",    Q2_loss_val,    global_step)
                writer.add_scalar("Metric/Policy_loss", policy_loss_val, global_step)
            except Exception as tb_err:
                print(f"train_sac: TensorBoard write failed: {tb_err}")
        else:
            print(f"train_sac: No updates performed this epoch (insufficient replay buffer data)")
            
        print(f"===== EPOCH {epoch}/{config.epochs} COMPLETED =====\n")
        schedule.step()

        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
            mem_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
            max_mem_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            print(
                f"CUDA Memory - Allocated: {mem_alloc:.2f} MB | "
                f"Reserved: {mem_reserved:.2f} MB | Max Allocated: {max_mem_alloc:.2f} MB"
            )
            torch.cuda.reset_peak_memory_stats(device)


        torch.cuda.empty_cache()



    # FINAL CHECKPOINT
    print(f"train_sac: Training completed! Saving final models...")
    torch.save(policy.state_dict(), log_dir / "policy.pt")
    torch.save(q1.state_dict(),      log_dir / "q1.pt")
    torch.save(q2.state_dict(),      log_dir / "q2.pt")
    print(f"train_sac: Final models saved to {log_dir}")
    print(f"train_sac: Final training stats - total_env_steps:{total_env_steps}, global_step:{global_step}, best_performance:{best_rews_mean:.3f}")
    writer.close()
    print(f"train_sac: TensorBoard writer closed. Training finished!")
    return float(best_rews_mean)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config JSON")
    args = parser.parse_args()
    cfg = Config.from_file(Path(args.config))
    train_sac(cfg)
