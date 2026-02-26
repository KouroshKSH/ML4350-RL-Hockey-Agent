# Overview

This document explains the workflow for environment setup, agent development, and training on the TCML cluster. Following these conventions ensures our code-base remains homogeneous and our experiments reproducible.

Be aware, as of writing this, the project (and its files) is in the following directory:

```
/home/stud.../rl_project/hockey_project/
```

You can replicate the same folder structure to avoid any future problems.

---

## 1. Environment Setup (Singularity)

We use **Singularity Sandboxes** (folders) instead of `.simg` files. This allows us to update packages instantly without rebuilding a massive image. It only requires the `requirements.txt` file to be present in the project's root directory.

### Initial Creation

To create the environment folder from the definition file (`hockey.def`), run:

```bash
singularity build --sandbox my_env_folder/ hockey.def
```

> This creates a writable folder named 'my_env_folder/'.

### Updating Packages

If you need to install a new library (e.g., `wandb`), do **not** rebuild the image. Use the writable shell:

First, enter the sandbox:

```bash
singularity shell --writable my_env_folder/
```

Once you're inside the Singularity shell, install your package (e.g. `numpy` for this example):

```bash
Singularity> pip install numpy
Singularity> python -c "import numpy; print('Success')" # Sanity check
Success # a printed message from the terminal
Singularity> exit
```

*Note: Changes made in the writable shell are persistent.*

Just to double check, go back to Singularity via the command below:

```bash
singularity shell my_env_folder/
```

There, retry printing the _success_ message to double-check.

---

## 2. Agent Development Convention

To maintain a clean repository, each algorithm must have its own python program (e.g. `td3_agent.py`). **Do _NOT_ use a shared python program for all agents!**

### Directory Structure

```text
hockey_project/
├── agent/               # Initial Agent Implementation
│   ├── td3_agent.py     # Example: TD3's core algorithm logic
│   ├── memory.py        # replay buffer
│   ├── networks.py      # actor/critic architectures
│   └── train.py         # training loop & WandB logging
```

### Consistency Rules

1. **Naming:** Use `agent_<ALGO_NAME>`. Inside, name the main class file `<algo>_agent.py`.
2. **Train.py:** Copy the `train.py` from `agent` as a template. Avoid changing the `wandb` initialization or the logging logic. Limit changes to the specific reinforcement learning logic to ensure our plots remain comparable.

---

## 3. Training on TCML

### Configuration & Keys

1. **WandB:** Create a folder named `private/` in the project root.
2. **API Key:** Place your Weights & Biases key in `private/wandb_api_key.txt`. The bash script reads this file to log you in automatically.

### Using the Training Template

Our master script is located at `scripts/template.sh`.

1. **Create your experiment script:**
```bash
cp scripts/template.sh scripts/td3_Abtin_v1.sh
chmod +x scripts/td3_Abtin_v1.sh

```


2. **Modify the variables:** Open your new script and update:
* `EMAIL`: Your university email for Slurm notifications.
* `AGENT`: The folder name (e.g., `agent_TD3`).
* `CONFIGPATH`: Path to your `.yaml` hyperparameter file.
* `MAXSEED`: How many parallel seeds to run.


3. **Sanity Check:** Before submitting to the cluster, run the script locally for a second to ensure there are no syntax errors in the bash logic:
```bash
./scripts/your_script_name.sh

```


4. **Submit:** If the sanity check passes, then pass the script to the server to run it automatically via `bash scripts/your_script_name.sh`.

---

## 4. Monitoring Results

### Log Files

Slurm logs are saved in the `logs/` folder using a specific timestamp format:
- `logs/yyyy-mm-dd_HH-MM-SS_seedN_jobID.log` (standard output)
- `logs/yyyy-mm-dd_HH-MM-SS_seedN_jobID.err` (errors and WandB info)

### Interpreting Progress

Check your `.log` files to see real-time game results:

```text
Job started at: Fri Feb  6 20:14:22 CET 2026
Running from: /home/stud376/rl_project/hockey_project
Seed 1 | Algorithm DDPG | Steps 250 | Reward -35.94 | Winner: 0
Seed 1 | Algorithm DDPG | Steps 2067 | Reward -12.02 | Winner: -1
...
Seed 1 | Algorithm DDPG | Steps 13493 | Reward 9.71 | Winner: 1
...
Seed 1 | Algorithm DDPG | Steps 142794 | Reward -3.18 | Winner: 0
Job finished at: Fri Feb  6 20:36:37 CET 2026
```

> **Winner Key:**
> * `+1`: Our agent won.
> * `0`: Tie / Draw.
> * `-1`: The basic opponent won.

You can also check the `.err` file in case there's any specific bug in the code (e.g. missing module, syntax error, etc.). A healthy run's file looks like this:

```text
wandb: [wandb.login()] Loaded credentials for https://api.wandb.ai from WANDB_API_KEY.
wandb: Currently logged in as: xxx to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
...
wandb: Run data is saved locally in /home/stud.../rl_project/hockey_project/wandb/run-...
...
wandb: ⭐️ View project at https://wandb.ai/...
wandb: 🚀 View run at https://wandb.ai/...
...
wandb: updating run metadata
wandb: uploading history steps 142141-143492, summary, console lines 108-108
wandb: 
wandb: Run history:
wandb:              Episode ▁▁▂▂▂▂▂▂▃▃▃▃▃▄▄▄▅▅▅▅▅▅▅▆▆▆▆▆▆▆▇▇▇▇▇▇████
wandb:           Loss/Actor ████▇▇▆▆▆▅▄▄▄▄▃▃▃▃▃▂▂▂▂▂▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁
wandb:          Loss/Critic ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▂▂▃▄▃▃▃▃▃▃▃▃▃▅▆▅▅▃▄▄▆█▅▅▄▄
wandb:         Reward/Total ▃▆▆▆▆▅▆▆█▆▇▆▆▃▆▆▆▇▆▄▇▆▆█▆▆█▆▅▆▆▇█▆▅▁▆▆▃▅
wandb: Stats/Episode_Length ██▁▃▂▂▂▁▂▁█▂▂█▂▁█▁▄███▂▁▂▄█▁██▁██▂██▃▂██
wandb:         Stats/Mean_Q ▁▁▁▁▂▂▃▃▃▃▃▄▃▄▄▅▅▅▆▆▇▇▇▇████████████████
wandb:   Stats/Success_Rate ▅▄▅▄▃█▇▇▇▆▁▁▁▂▂▂▃▂▄▅▄▂▄▃▂▄▄▃▄▄▅▆▄▄▄▄▅▅▇▇
wandb: 
wandb: Run summary:
wandb:              Episode 999
wandb:           Loss/Actor -66.36252
wandb:          Loss/Critic 9.999
wandb:         Reward/Total -24.93129
wandb: Stats/Episode_Length 249
wandb:         Stats/Mean_Q 64.82987
wandb:   Stats/Success_Rate 0.22
wandb: 
wandb: 🚀 View run 2026-02-06_20-14-31_Kourosh_seed1 at: https://wandb.ai/...
wandb: ⭐️ View project at: https://wandb.ai/...
wandb: Synced 5 W&B file(s), 100 media file(s), 0 artifact file(s) and 9 other file(s)
wandb: Find logs at: ./wandb/run-.../logs
```

### WandB Integration

Check the `.err` file for the WandB URL. You can track the `Success_Rate`, `Mean_Q`, and `Loss` curves live in the browser. Or open the workspace with [this link](https://wandb.ai/kourosh-sharifi-tuebingen-university/RL-hockey?nw=nwuserkouroshsharifitue).

**Note:** For more details on cluster resource limits (CPU/Memory), please refer to the comments in **Pull Request #1**, found [on GitHub](https://github.com/KouroshKSH/RL-project-temp/pull/1).