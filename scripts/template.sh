#!/bin/bash

# --- Configuration ---
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)

# use absolute paths to avoid any "file not found" errors on compute nodes
# change the student ID when needed
STUDENTID="stud376"

# This automatically finds the project root regardless of who is running it
PROJ_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# singularity can run a folder just like an '.simg' file
IMAGE_PATH="${PROJ_DIR}/my_env_folder"

# change this accordingly
EMAIL="abtin.mogharabin@student.uni-tuebingen.de"
# or set it as
# EMAIL="kourosh.sharifi@student.uni-tuebingen.de"

# make sure this is set to the exact config file you want (hyperparams, algo, etc.)
CONFIGPATH="configs/ddpg_basic.yaml"

# change the agent's folder per algorithm
AGENT="agent"

# if the script has to run for more than hour, change from here
TIMELIMIT="01:00:00"

# if you need more CPUs per task, increase this
CPUPERTASK="2"

# if you need more memory (in GB), increase this
MEMORY="8"

# increase this to run for more seeds per experiment
MAXSEED=3

# --- Security: Read W&B Key from private folder ---
if [ -f "${PROJ_DIR}/private/wandb_api_key.txt" ]; then
    export WANDB_API_KEY=$(cat "${PROJ_DIR}/private/wandb_api_key.txt")
    export WANDB_MODE="online"
else
    echo "Warning: W&B key not found. Running in offline mode."
    export WANDB_MODE="offline"
fi

# --- Start training ---
for (( SEED=1; SEED<=MAXSEED; SEED++ ))
do
  sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=hockey_s${SEED}
#SBATCH --partition=day
#SBATCH --time=${TIMELIMIT}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=${CPUPERTASK}
#SBATCH --mem=${MEMORY}
#SBATCH --output=${PROJ_DIR}/logs/${TIMESTAMP}_seed${SEED}_%j.log
#SBATCH --error=${PROJ_DIR}/logs/${TIMESTAMP}_seed${SEED}_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=${EMAIL}

# 1. Enter the project directory
cd ${PROJ_DIR}

# 2. Create directories explicitly
mkdir -p logs runs checkpoints

echo "Job started at: \$(date)"
echo "Running from: \$(pwd)"

# 3. Execute using the absolute path to the image
singularity exec --nv ${IMAGE_PATH} \
    python3 ${AGENT}/train.py --config ${CONFIGPATH}

echo "Job finished at: \$(date)"
EOT
done