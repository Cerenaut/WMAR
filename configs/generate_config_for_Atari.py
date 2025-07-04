import json
import os
import copy

# Hard-coded base configuration template with six environments
base_config = {
    "esc": {
        "env_schedule_type": None,  
        "env_configs": [
            {"name": "ALE/MsPacman-v5",     "kwargs": {}, "rew_scale": 0.05},
            {"name": "ALE/Boxing-v5",       "kwargs": {}, "rew_scale": 1},
            {"name": "ALE/CrazyClimber-v5", "kwargs": {}, "rew_scale": 0.001},
            {"name": "ALE/Frostbite-v5",    "kwargs": {}, "rew_scale": 0.2},
            {"name": "ALE/Seaquest-v5",     "kwargs": {}, "rew_scale": 0.5},   # added
            {"name": "ALE/Enduro-v5",       "kwargs": {}, "rew_scale": 0.5}    # added
        ],
        "kwargs": {}
    },
    "algorithm": None,
    "sac_lr": 0.0003,
    "sac_batch_size": 256,
    "sac_dv3_data_n_max": 1024,
    "sac_tau": 0.005,
    "sac_gamma": 0.99,
    "sac_alpha": 0.2,
    "img_size": 64,
    "seed": None,
    "epochs": None,  # overridden per case below
    "wm_lr": 0.0001,
    "log_frequency": 1000,
    "steps_per_batch": 1000,
    "ac_train_steps": 800,
    "ac_train_sync": 128,
    "fresh_ac": False,
    "n_sync": 4,
    "gen_seq_len": 4096,
    "env_repeat": 4,
    "data_n": 32,
    "data_n_max": 512,
    "data_t": 512,
    "mb_t_size": 32,
    "mb_n_size": 16,
    "random_policy": "first",
    "pretrain_enabled": False,
    "pretrain_data_multiplier": 4,
    "pretrain_mb_t_size": 8,
    "pretrain_mb_n_size": 16,
    "pretrain_steps": 30000,
    "gru_units": 512,
    "cnn_depth": 32,
    "mlp_features": 512,
    "mlp_layers": 2,
    "wall_time_optimisation": False,
    "action_space": 18,
    "replay_buffers": [
        {"rb_type": "FifoReplay",     "rb_device": "cuda"},
        {"rb_type": "LongTermReplay", "rb_device": "cuda"}
    ]
}

algorithms = ["sac", "dv3", "wmar"]
seeds = [1337, 31337, 42, 987654321, 123456789]
output_root = "configs"

for alg in algorithms:
    alg_dir = os.path.join(output_root, alg)
    os.makedirs(alg_dir, exist_ok=True)

    # SINGLE-ENV configs (one env per file)
    for env_idx, single_env in enumerate(base_config["esc"]["env_configs"]):
        for seed in seeds:
            cfg = copy.deepcopy(base_config)
            cfg["algorithm"] = alg
            cfg["seed"] = seed

            cfg["esc"]["env_schedule_type"] = "AllEnvironments"
            cfg["esc"]["env_configs"] = [single_env]
            cfg["esc"]["kwargs"] = {}
            cfg["epochs"] = 91

            if alg != "wmar":
                cfg["replay_buffers"] = [cfg["replay_buffers"][0]]

            fname = f"config_seed{seed}_env{env_idx}.json"
            with open(os.path.join(alg_dir, fname), "w") as f:
                json.dump(cfg, f, indent=4)

    # MULTI-ENV configs (all envs in one file)
    for seed in seeds:
        cfg = copy.deepcopy(base_config)
        cfg["algorithm"] = alg
        cfg["seed"] = seed

        cfg["esc"]["env_schedule_type"] = "SequentialEnvironments"
        cfg["esc"]["env_configs"] = copy.deepcopy(base_config["esc"]["env_configs"])
        cfg["esc"]["kwargs"] = {"swap_sched": 90}
        cfg["epochs"] = 361

        if alg != "wmar":
            cfg["replay_buffers"] = [cfg["replay_buffers"][0]]

        fname = f"config_seed{seed}.json"
        with open(os.path.join(alg_dir, fname), "w") as f:
            json.dump(cfg, f, indent=4)

print("Done! Generated 90 single-env + 15 multi-env = 105 files with six environments.")
