#!/bin/bash

ALGO="TQC"

# --- Configuration ---
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# use absolute paths to avoid any "file not found" errors on compute nodes
# change the student ID when needed
STUDENTID="stud376"
PROJ_DIR="/home/${STUDENTID}/rl_project/hockey_project"

# singularity can run a folder just like an '.simg' file
IMAGE_PATH="${PROJ_DIR}/my_env_folder"

# can be changed to abtin.mogharabin@student.uni-tuebingen.de
EMAIL="kourosh.sharifi@student.uni-tuebingen.de"

# make sure this is set to the exact config file you want (hyperparams, algo, etc.)
CONFIGPATH="configs/tqc_leaderboard.yaml"

# change the agent's folder per algorithm
AGENT="agent"

# Partition and Time: Using 'week' for a 10k episode TQC run
PARTITION="week"
TIMELIMIT="3-00:00:00" # 3 days is plenty for 10k episodes + League overhead
CPUPERTASK="4"
MEMORY="8G"
MAXSEED=1

# --- Security: Read W&B Key ---
if [ -f "${PROJ_DIR}/private/wandb_api_key.txt" ]; then
    export WANDB_API_KEY=$(cat "${PROJ_DIR}/private/wandb_api_key.txt")
    export WANDB_MODE="online"
else
    export WANDB_MODE="offline"
fi

# --- Start training ---
for (( SEED=1; SEED<=MAXSEED; SEED++ ))
do
  # Calculate a unique seed based on the loop
  CURRENT_SEED=$((42 + SEED))
  
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=${ALGO}_s${CURRENT_SEED}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIMELIMIT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUPERTASK}
#SBATCH --mem=${MEMORY}
#SBATCH --output=${PROJ_DIR}/logs/${TIMESTAMP}_seed${CURRENT_SEED}_%j.log
#SBATCH --error=${PROJ_DIR}/logs/${TIMESTAMP}_seed${CURRENT_SEED}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${EMAIL}

cd ${PROJ_DIR}
mkdir -p logs checkpoints results

# Execute TQC Training
singularity exec --nv ${IMAGE_PATH} \
    python3 ${AGENT}/train.py --config ${CONFIGPATH} --seed ${CURRENT_SEED} --user Kourosh
EOT
done