import time
from datetime import datetime
import argparse
from pathlib import Path
from typing import Optional
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
    start_time = datetime.now()
    print(f"Training started at {start_time.isoformat()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file")
    save_nets = False
    log_dir = None
    log_images = False
    
    args = parser.parse_args()
    if args.config is not None:
        config = Config.from_file(Path(args.config))
    else:
        config = None

    if config.algorithm == "sac":
        # dispatch into sac.py, skip all WMAR/DV3 setup
        import sac
        sac.train_sac(config)
        exit(0)
    
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
        log_dir  = os.path.join("runs", config.algorithm, run_name)      


    writer = SummaryWriter(log_dir=log_dir)
    log_dir = Path(log_dir)
    config.save(log_dir / "config.json")

    
    total_env_steps = 0        # number of *real* environment interactions so far

    best_rews_mean = float("-inf")
    global_step = 0            # gradient updates so far  training iterations
    runs_root = Path("runs") / config.algorithm  
    runs_root.mkdir(parents=True, exist_ok=True)

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

            # Each tuple (t, n) counts as one env step; multiply by env_repeat.

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
            print("Evaluation started ...")
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

            writer.add_scalar("Sample/total_train_iters", global_step, global_step)


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
            global_step += 1  # training‑iteration counter

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

    end_time = datetime.now()
    print(f"Training ended   at {end_time.isoformat()}")

    # Compute duration
    duration = end_time - start_time

    # Append to log file
    log_line = (
        f"algorithm: {config.algorithm}\n"
        f"seed: {config.seed}\n"
        f"Num of epochs: {config.epochs}\n"
        f"Start:    {start_time.isoformat()}\n"
        f"End:      {end_time.isoformat()}\n"
        f"Duration: {duration}\n"
        "----------------------------------------\n"
    )
    with open("training_times.txt", "a") as log_file:
        log_file.write(log_line)

