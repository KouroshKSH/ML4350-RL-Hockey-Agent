#!/bin/bash
# internal_leaderboard/run_tournament.sh

set -e

STUDENTID="stud376" # put your student ID here
PROJ_DIR="/home/${STUDENTID}/rl_project/hockey_project"
SCRIPT_DIR="${PROJ_DIR}/internal_leaderboard"
IMAGE_PATH="${PROJ_DIR}/my_env_folder"

# --- Security: Read W&B Key ---
if [ -f "${PROJ_DIR}/private/wandb_api_key.txt" ]; then
    export WANDB_API_KEY=$(cat "${PROJ_DIR}/private/wandb_api_key.txt")
    export WANDB_MODE="online"
else
    export WANDB_MODE="offline"
fi

cd "${SCRIPT_DIR}"

echo "Running tournament from: ${SCRIPT_DIR}"

# Explicitly bind /home so Singularity always sees your files
singularity exec --nv \
    --bind /home:/home \
    "${IMAGE_PATH}" \
    python3 tournament.py
