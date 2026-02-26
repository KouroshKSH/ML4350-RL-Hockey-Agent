#!/bin/bash
#SBATCH --job-name=hockey_sweep3000Grid
#SBATCH --partition=week
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/sweep3000_%j.log
#SBATCH --error=logs/sweep3000_%j.err

set -e

PROJ_DIR="$HOME/rl_project/hockey_project"
IMAGE_PATH="${PROJ_DIR}/my_env_folder"

cd "${PROJ_DIR}"
mkdir -p logs results sweeps

# W&B key handling
if [ -f "${PROJ_DIR}/private/wandb_api_key.txt" ]; then
    export WANDB_API_KEY=$(cat "${PROJ_DIR}/private/wandb_api_key.txt")
    export WANDB_MODE="online"
else
    export WANDB_MODE="offline"
fi

singularity exec --nv --bind /home:/home "${IMAGE_PATH}" \
  python3 scripts/small_grid_search.py \
    --algos TQC SAC REDQ DROQ CROSSQ TD3 \
    --episodes 3000 \
    --seeds 43 44 45 \
    --eval_interval 150 \
    --eval_eps_per_opp 10 \
    --eval_opponents weak strong self_play \
    --user_prefix sweep3000
