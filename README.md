# ARROW


## Abstract
Continual reinforcement learning challenges agents to acquire new skills while retaining
previously learned ones with the goal of improving performance across both past and future
tasks. Most existing approaches rely on model-free methods with replay buffers to mitigate
catastrophic forgetting, however these solutions often face significant scalability challenges
due to large memory demands. Drawing inspiration from neuroscience, where the brain
replays experiences to a predictive world model rather than directly to the policy, we present
ARROW (Augmented Replay for RObust World models), a model-based continual RL
algorithm that extends DreamerV3 with a memory-efficient, distribution-matching replay
buffer. Unlike standard fixed-size FIFO buffers, ARROW maintains two complementary
buffers: a short term buffer for recent experiences and a long-term buffer that preserves the
task diversity by sampling intelligently. We evaluate ARROW on two challenging continual
RL settings: Tasks without shared structure (Atari), and tasks with shared structure, where
knowledge transfer is possible (Procgen Coinrun variants). Compared to model-free and
model-based baselines with same-size replay buffers, ARROW demonstrates substantially
reduced forgetting on tasks without shared structure tasks, while maintaining comparable
forward transfer. Our findings highlight the potential of model-based RL and bio-inspired
approaches for continual reinforcement learning justifying further research.


## Repository Layout
- `Code/`
  - `ARROW_and_DV3/`
    - `Atari/`
      - `train.py` — entrypoint for ARROW/DV3 runs.
      - `ac.py`, `wm.py`, `rssm.py`, `vae.py`, `replay.py`, `config.py`, `generate_trajectory.py` — actor-critic, world model, latent dynamics, VAE, replay buffer, default hyperparameters, and trajectory generation utilities.
    - `CoinRun/` — same file set as Atari, tailored for Procgen CoinRun.
  - `SAC/`
    - `Atari/`
      - `sac.py` — SAC entrypoint.
      - `ac.py`, `rssm.py`, `vae.py`, `replay.py`, `config.py`, `generate_trajectory.py` — SAC components and helpers.
    - `CoinRun/` — same file set as Atari, tailored for Procgen CoinRun.
- `Configs/`
  - `Atari configs/` - Note: all config files named with task (game), task id `e*`, seed `s*`, and method (ARROW, DV3, and SAC).
    - `CL-task configs/` — continual-learning configs for ARROW, DV3, and SAC.
    - `Single-task configs/` — per-game Atari configs for ARROW, DV3, and SAC.
  - `ConRun configs/`
    - `CL-task configs/` — same file set as Atari, tailored for Procgen CoinRun.
    - `Single-task configs/` — same file set as Atari, tailored for Procgen CoinRun.
- `requirements.txt` — Python dependencies.

## Environment Setup
```bash
# Load conda into the current shell
source ~/miniconda3/etc/profile.d/conda.sh

# Create and activate environment
conda create -n arrow python=3.10 -y
conda activate arrow

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```
- The `requirements.txt` pins CUDA 11.8 wheels for PyTorch; adjust the `--extra-index-url` or versions if your CUDA/toolkit differs.

## Running Locally
- `cd path/to/ARROW/Folder`
- ARROW and DV3: `python Code/ARROW_and_DV3/Atari/train.py --config /path/to/config.json`
- SAC: `Code/SAC/Atari/sac.py --config /path/to/config.json`

## SLURM Usage (example)
The jobs are typically run as arrays, one config per task id:
```bash
#!/bin/bash
#SBATCH --job-name=arrow_atari
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1
#SBATCH --time=12:00:00
#SBATCH --mem=45G
#SBATCH --array=0-4
#SBATCH --output=logs/arrow/%A_%a.txt

# Load conda initialization
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arrow

# Move to project directory (example)
cd "/path/to/ARROW/" 

# Collect config files (sorted)
mapfile -t configs < <(ls "Configs/Atari configs/Single-task configs"/*.json | sort)
config_file="${configs[$SLURM_ARRAY_TASK_ID]}"

echo "[$SLURM_ARRAY_TASK_ID] $(date) → $config_file"

# Run experiment
    python Code/ARROW_and_DV3/Atari/train.py --config "$config_file"
# or for sac
#   python Code/SAC/Atari/sac.py --config "$config_file"
```


## Tips
- Adjust SLURM directives (partition, GPU type, memory, time) to your cluster (if you're using SLURM)
- Each config file contains the sequence of tasks (or single task), seed, and method.
- Keep configs sorted and align seeds across methods to simplify comparisons.
- Log files in `logs/` will follow the `%A_%a` pattern; ensure the directory exists before launching jobs.
