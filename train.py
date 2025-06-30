import argparse
from pathlib import Path
from typing import Optional
# This is how tensorboard names the folder
import os
import socket
from datetime import datetime
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import replay
from ac import ActorCriticOpt, train_ac_from_wm
from config import Config, EnvConfig, EnvScheduleConfig, RbConfig
from generate_trajectory import (
    SequentialEnvironments,
    evaluate,
    generate_trajectories,
    reinterpret_nt_to_t_n,
)
from wm import WorldModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file")
    parser.add_argument("--seed", type=int, help="RNG seed (overrides config)")
    parser.add_argument("--agent", help="Algorithm variant: 'wmar' (augmented replay) or 'dv3' (FIFO-only)")
    
    args = parser.parse_args()
    if args.config is not None:
        config = Config.from_file(Path(args.config))
    else:
        config = None

    save_nets = False
    log_dir = None
    log_images = False

    default_config = Config(
        esc=EnvScheduleConfig(
            env_schedule_type=SequentialEnvironments,
            env_configs=[
                EnvConfig("CoinRun"),
                EnvConfig("CoinRun+NB"),
                EnvConfig("CoinRun+NB+RT"),
                EnvConfig("CoinRun+NB+RT+MA"),
                EnvConfig("CoinRun+NB+RT+MA+UGA"),
                EnvConfig("CoinRun+NB+RT+MA+UGA+CA"),
            ],
            kwargs={"swap_sched": 90},
        ),
        seed=1337,
        epochs=361,
        wm_lr=1e-4,
        log_frequency=1000,
        steps_per_batch=1000,
        ac_train_steps=800,
        ac_train_sync=128,
        fresh_ac=False,
        n_sync=4,
        gen_seq_len=4096,
        env_repeat=1,
        data_n=32,
        data_n_max=512,
        data_t=512,
        mb_t_size=32,
        mb_n_size=16,
        random_policy="first",
        pretrain_enabled=False,
        pretrain_data_multiplier=4,
        pretrain_mb_t_size=8,
        pretrain_mb_n_size=16,
        pretrain_steps=30_000,
        gru_units=512,
        cnn_depth=32,
        mlp_features=512,
        mlp_layers=2,
        wall_time_optimisation=False,
        action_space=15,
        replay_buffers=[
            RbConfig(replay.FifoReplay, "cuda"),#default is for dv3
        ],
    )
     # ---------------------------------------------------------------------
    # Override replay buffer choice based on --agent -----------------------
    # ---------------------------------------------------------------------

    config = config if config is not None else default_config

    if args.seed is not None:
            config.seed = args.seed
    
    running = "dv3"
    if args.agent.lower() != "dv3":
        running = "WMAR"
        config.replay_buffers.append(RbConfig(replay.LongTermReplay, "cuda")) #for WMAR

    print("replay_buffers:")
    print(len(config.replay_buffers))
    print(config.replay_buffers)

    
    torch.random.manual_seed(config.seed)
    np.random.seed(config.seed)
    print("Training with seed: ", config.seed)
    wm = WorldModel(
        3,
        (32, 32),
        config.action_space,
        config.gru_units,
        config.cnn_depth,
        config.mlp_features,
        config.mlp_layers,
        config.wall_time_optimisation,
    ).cuda()
    opt = Adam(wm.parameters(), lr=config.wm_lr)

    envs = config.get_env_schedule()
    replay = config.get_replay_buffer()

    # OPTIONAL: Load from existing
    aco: Optional[ActorCriticOpt] = None

    if not log_dir:

        current_time = datetime.now().strftime("%b%d_%H-%M-%S")
        run_name = f"{current_time}_{socket.gethostname()}_seed{config.seed}"
        # put files in runs/<running>/<run_name>/
        log_dir  = os.path.join("runs", running, run_name)      


    writer = SummaryWriter(log_dir=log_dir)
    log_dir = Path(log_dir)
    config.save(log_dir / "config.json")

    # -------------------------------------------------------------------------
    # NEW METRIC STATE: sample‑efficiency bookkeeping
    # -------------------------------------------------------------------------
    total_env_steps = 0        # number of *real* environment interactions so far
    # (training iterations == global_step; declared later)

    best_rews_mean = float("-inf")
    global_step = 0            # gradient updates so far  ≙  training iterations
    runs_root = Path("runs") / running          # running == "WMAR" or "DV3'"
    runs_root.mkdir(parents=True, exist_ok=True)
    epoch_file = runs_root / "current_epoch.txt"

    for epoch in range(config.epochs):

        if config.random_policy == "first":
            random_policy = epoch == 0
        elif config.random_policy == "new":
            random_policy = envs.is_new_env()
        for _ in range(
            config.pretrain_data_multiplier if random_policy and config.pretrain_enabled else 1
        ):
            _acts, _obss, _rews, _conts, _resets = reinterpret_nt_to_t_n(
                *generate_trajectories(
                    config.n_sync * config.gen_seq_len,
                    config.n_sync,
                    wm=wm,
                    ac=None if random_policy else aco.ac,
                    env_fns=envs.funcs(),
                    env_repeat=config.env_repeat,
                ),
                config.data_t,
                config.data_n,
            )
            replay.add(_acts, _obss, _rews, _conts, _resets)

            # -----------------------------------------------------------------
            # *Sample‑efficiency metric 1*: real env experiences gathered so far
            # Each tuple (t, n) counts as one env step; multiply by env_repeat.
            # -----------------------------------------------------------------
            num_new_env_steps = _acts.shape[0] * _acts.shape[1] * config.env_repeat
            total_env_steps += num_new_env_steps
            writer.add_scalar("Sample/total_env_steps", total_env_steps, global_step)
            # Also expose current replay size – handy to visualise replay reuse
            writer.add_scalar("Sample/replay_buffer_size", replay.n_valid, global_step)

        envs.step()
        print(f"{replay.n_valid=}")

        rews_eps_mean = _rews.sum().item() / _resets.sum().item()
        writer.add_scalar("Perf/rews_eps_mean", rews_eps_mean, global_step)
        len_eps_mean = config.gen_seq_len / _resets.sum().item() * config.env_repeat
        writer.add_scalar("Perf/len_eps_mean", len_eps_mean, global_step)
        if rews_eps_mean >= best_rews_mean:
            best_rews_mean = rews_eps_mean
            if save_nets and aco is not None:
                print(f"Saving best rews eps mean {rews_eps_mean=}")
                torch.save(wm.state_dict(), log_dir / "save_wm_best.pt")
                torch.save(aco.ac.state_dict(), log_dir / "save_ac_best.pt")

        # Evaluation games
        if epoch % 10 == 0:
            eval_results_mean = []
            eval_results_std = []
            eval_funcs = envs.eval_funcs()
            for env_fns in eval_funcs:
                ev_eps_mean, ev_eps_std = evaluate(
                    config.n_sync,
                    wm=wm,
                    ac=aco.ac if aco is not None else aco,
                    env_fns=env_fns,
                    env_repeat=config.env_repeat,
                    n_rollouts=256,
                )
                eval_results_mean.append(ev_eps_mean)
                eval_results_std.append(ev_eps_std)
            writer.add_scalars(
                "Perf/eval_rew_eps_mean",
                {f"{i}": m for i, m in enumerate(eval_results_mean)},
                global_step,
            )
            writer.add_scalars(
                "Perf/eval_rew_eps_std",
                {f"{i}": s for i, s in enumerate(eval_results_std)},
                global_step,
            )

        envs.step()

        progbar = trange(
            config.steps_per_batch
            if epoch > 0 or not config.pretrain_enabled
            else config.pretrain_steps,
            desc=f"Epoch {epoch + 1}/{config.epochs}",
        )
        for _ in progbar:
            if epoch > 0 or not config.pretrain_enabled:
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets = replay.minibatch(
                    config.mb_t_size, config.mb_n_size
                )
            else:
                mb_acts, mb_obss, mb_rews, mb_conts, mb_resets = replay.minibatch(
                    config.pretrain_mb_t_size, config.pretrain_mb_n_size
                )

            loss, metrics = wm.compute_loss(mb_acts, mb_obss, mb_rews, mb_conts, mb_resets)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1000)
            opt.step()

            # -------------------------------------------------------------
            # *Sample‑efficiency metric 2*: number of training iterations
            # (we simply mirror global_step so it appears explicitly)
            # -------------------------------------------------------------
            writer.add_scalar("Sample/total_train_iters", global_step, global_step)

            # Optional progress bar logging
            # if global_step % 10 == 0:
            #     progbar.set_postfix({k: f"{v:.2f}" for k, v in metrics.items()})

            if global_step % config.log_frequency == 0:
                writer.add_scalar("Metric/grad_norm", grad_norm, global_step)
                with torch.no_grad():
                    for metric_key, metric_value in metrics.items():
                        writer.add_scalar(metric_key, metric_value, global_step)

                    if log_images:
                        original = _obss[:16, 0:2].cuda()
                        writer.add_images(
                            "original", original.swapaxes(0, 1).flatten(0, 1), global_step
                        )

                        init_z, init_h = wm.rssm.initial_state(original.shape[1])
                        no_resets = torch.zeros(*original.shape[:2], 1, device=init_z.device)
                        z_posts, z, h = wm.rssm(
                            init_z, _acts[:, 0:2].cuda(), init_h, original, no_resets
                        )
                        zhs = wm.zh_transform(z, h)
                        recon = torch.stack([wm.decoder(zh) for zh in zhs])

                        writer.add_images(
                            "reconstructed",
                            recon.clip(0, 1).swapaxes(0, 1).flatten(0, 1),
                            global_step,
                        )
                        writer.add_images(
                            "latent",
                            z_posts.exp().swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
                        writer.add_images(
                            "latent sample",
                            z.swapaxes(0, 1).flatten(0, 1).unsqueeze(1),
                            global_step,
                        )
            global_step += 1  # ----------------------------------  training‑iteration counter

        if config.fresh_ac and epoch % config.fresh_ac == 0:
            aco, approx_perf = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                config.ac_train_sync,
                dream_steps=16,
                lr=4e-4,
            )
        else:
            aco, approx_perf = train_ac_from_wm(
                wm,
                replay,
                config.ac_train_steps,
                config.ac_train_sync,
                dream_steps=16,
                aco=aco,
                lr=1e-4,
            )

        writer.add_scalar("Perf/approx_perf", approx_perf, global_step)

        if save_nets:
            torch.save(wm.state_dict(), log_dir / "save_wm.pt")
            torch.save(aco.ac.state_dict(), log_dir / "save_ac.pt")
        
        torch.cuda.empty_cache()

