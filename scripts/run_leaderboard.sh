#!/bin/bash
export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN="..." # private token (generated per account)

STUDENTID="..." # student ID (assigned per person)
PROJ_DIR="/home/${STUDENTID}/rl_project/hockey_project"
IMAGE_PATH="${PROJ_DIR}/my_env_folder"

# Use absolute path for the script
cd ${PROJ_DIR}

# Run the client through Singularity
singularity exec ${IMAGE_PATH} python3 agent/leaderboard_agent.py