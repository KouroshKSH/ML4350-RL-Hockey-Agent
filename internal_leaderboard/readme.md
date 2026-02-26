# Internal Leaderboard & Tournament

This directory contains the scripts to rank agents trained during the development of the whole project (upwards of 1,000 as of writing this). It uses a Round-Robin format to ensure every model is tested against every other model.

### Contents
* `tournament.py`: The core orchestrator that loads models and runs matches.
* `run_tournament.sh`: The entry-point script for running the tournament on the TCML cluster.
* `tournament_config.yaml`: Configuration for worker counts and trajectory saving.
* `leaderboards/`: Contains exported `.csv` files of tournament results.
* `trajectories/`: Stores `.pkl` files of game states (if enabled) for visualization.

### How to Run
1. **Configure:** Make sure your models are in the `results/` folder and your W&B key is in `private/`.
2. **Execute:** Run the bash script from the project root:
   ```bash
   bash internal_leaderboard/run_tournament.sh
   ```
3. **Monitor:** Real-time rankings and win/loss/tie stats will be uploaded to your WandB project.

---

### Step-by-Step: Running the Tournament
To run this on the TCML server using your Singularity sandbox:

1.  **Verify Paths:** Open `internal_leaderboard/run_tournament.sh` and ensure the `STUDENTID` and `IMAGE_PATH` variables match your current setup.
2.  **Set Permissions:** Make the script executable: 
    `chmod +x internal_leaderboard/run_tournament.sh`
3.  **Launch:** Execute the script: 
    `./internal_leaderboard/run_tournament.sh`
4.  **Automatic Login:** The script will automatically look for your `wandb_api_key.txt` to log you in. If it's missing, it will default to `offline` mode.
5.  **Check Results:** Once finished, look in `internal_leaderboard/leaderboards/` for the final CSV or check your WandB dashboard link.

