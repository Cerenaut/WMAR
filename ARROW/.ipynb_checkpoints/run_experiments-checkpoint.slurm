#!/bin/bash
#SBATCH --job-name=CoinRunWMAR
#SBATCH --partition=gpu           # the GPU queue (A40s)
#SBATCH --gres=gpu:A100:1          # one 40 GB A40
#SBATCH --time=3-00:00:00         # up to 3 days
#SBATCH --mem=48G                 # 48 GB RAM
#SBATCH --array=0-104             # 0–104 → 105 jobs total
#SBATCH --output=logs/slurm_%A_%a.txt

module purge
module load cuda/11.6
module load miniforge3
conda activate Aziz

cd ~/pc88/Aziz/

echo "Started at $(date)"

#—— map SLURM_ARRAY_TASK_ID → {algorithm, config file} ——
algorithms=(wmar sac dv3)
configs_per_algo=35

# which algorithm group?
algo_idx=$(( SLURM_ARRAY_TASK_ID / configs_per_algo ))
algo=${algorithms[$algo_idx]}

# which file within that group?
config_idx=$(( SLURM_ARRAY_TASK_ID % configs_per_algo ))
# sort to get a stable order, then pick line N (1-based)
config_file=$(ls configs/$algo/*.json | sort | sed -n "$((config_idx+1))p")

echo "Running $algo → $config_file"
python train.py --config "$config_file"

echo "Finished at $(date)"
nvidia-smi --query-gpu=name,index,memory.total --format=csv
