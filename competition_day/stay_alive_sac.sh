#!/bin/bash

# configs
ALGO="sac"
MODEL="/hockey_project/results/SAC/hockey/2026-02-25_Kourosh_comp/checkpoints/SAC__mvqo7ch5__phase3_league_ucb_cover__tot35000__seed44__late__ep20600.pth"
NORM="/hockey_project/results/SAC/hockey/2026-02-25_Kourosh_comp/checkpoints/SAC__mvqo7ch5__phase3_league_ucb_cover__tot35000__seed44__late__ep20600.npz"
TOKEN="..." # private token
URL="comprl.cs.uni-tuebingen.de"
PORT="65335"

# autostarting because of random server shutdowns
THRESHOLD_TIME_WINDOW=600  # 10 mins
THRESHOLD=10
termination_times=()

while true; do
    echo "##############################################"
    echo "#   Starting CompRL Unified Client Agent     #"
    echo "##############################################"

    # run the singularity command inside the loop
    singularity exec --nv ./my_env_folder \
      env \
        UNIFIED_ALGO="$ALGO" \
        UNIFIED_MODEL_PATH="$MODEL" \
        UNIFIED_NORM_PATH="$NORM" \
        COMPRL_SERVER_URL="$URL" \
        COMPRL_SERVER_PORT="$PORT" \
        COMPRL_ACCESS_TOKEN="$TOKEN" \
      python3 run_comprl_unified_client.py

    # track the crashes to prevent infinite loops if your path is wrong
    current_time=$(date +%s)
    termination_times+=("$current_time")
    
    # for old timestamps
    termination_times=($(for time in "${termination_times[@]}"; do
        if (( current_time - time <= THRESHOLD_TIME_WINDOW )); then
            echo "$time"
        fi
    done))

    if (( ${#termination_times[@]} > THRESHOLD )); then
        echo "Error: Too many crashes in 10 minutes. Check your paths/code!"
        exit 1
    fi

    echo "Disconnected from server. Reconnecting in 20 seconds..."
    sleep 20
done
