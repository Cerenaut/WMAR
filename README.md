# Tasks with shared structure

## Directory 

```
./
├── train.py                   # Unified entry point for DV3/WMAR/SAC
├── sac.py                     # SAC baseline implementation
├── ac.py                      # ActorCritic utilities for WMAR
├── wm.py                      # WorldModel implementation (WMAR)
├── vae.py                     # VAE encoder/decoder module
├── rssm.py                    # Recurrent State-Space Model (RSSM) core
├── replay.py                  # Replay buffers (FifoReplay, LongTermReplay)
├── config.py                  # Config dataclass + EnvScheduleConfig
├── generate_trajectory.py     # Env wrappers & data collection
└── runs/                      # TensorBoard logs & checkpoints
    ├── dv3/<run_name>/
    ├── wmar/<run_name>/
    └── sac/<run_name>/
```

Each `<run_name>` is automatically generated as:

```
<MonDD_HH-MM-SS>_<hostname>_seed<seed>
```

And includes:

- `config.json` (snapshot of hyperparameters)
- TensorBoard logs
- best & final model checkpoints

---

## Running an Algorithm

All runs use the same `train.py` script. You must use a config file with `--config`. For example:

```bash
python train.py --config "config.json"
```

