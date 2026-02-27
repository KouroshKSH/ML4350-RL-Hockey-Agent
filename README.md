# ShaBin: Laser Hockey Reinforcement Learning

This repository contains our contribution to the Reinforcement Learning Tournament of the University of Tübingen (Winter Term 2025/26) for the course **ML4350**, presented by [Prof. Dr. Georg Martius](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/team/prof-dr-georg-martius/).

![Laser Hockey Gameplay](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/misc/example_gameplay.gif)

> An [example gameplay](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/misc/example_gameplay.gif) of our agent against its opponent.

Our project focuses on training hockey agents to play the [Laser Hockey](https://github.com/martius-lab/hockey-env) game using off-policy algorithms and multi-agent training system.

## Tournament Results
Out of the competing teams, our solution was awarded **5th place**. 

| Algorithm | Author |
| :--- | :--- |
| [TQC](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/agent/tqc_agent.py) | Abtin Mogharabin |
| [SAC](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/agent/sac_agent.py) | Kourosh Sharifi |

> The official certificate for this achievement will be available [here]().

## Overview
- **Algorithms:** Implementation of Truncated Quantile Critics (TQC) and Soft Actor-Critic (SAC).
- **Reward Design:** Potential-based reward shaping, kinematic progress weights, and action $L_2$ regularization.
- **Self-Play:** Opponent sampling via Prioritized Fictitious Self-Play (PFSP) and Upper Confidence Bound (UCB).
- **League Training:** A "lite" version of Policy-Space Response Oracles (PSRO) using Replicator Dynamics to ensure meta-strategy stability.

![Project Diagram](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/misc/RL_project_diagram.png)

> A high-level [design](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/misc/RL_project_diagram.png) of the project's structure.

## Project Structure
Detailed information regarding the algorithms, ablation studies, and training results can be found in our **[Final Project Report](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/misc/RL_2025-26_Final_Project_Report_ShaBin.pdf)**.

If you wish to replicate our environment setup or training pipeline on a cluster (TCML), please refer to the [instruction file](https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent/blob/master/instructions.md) named `instructions.md`.

## Citation
If you find this work useful for your research or projects, please consider citing it:

```latex
@proceedings{shabin2026hockey,
  title  = {ShaBin: Competitive Laser Hockey via Maximum Entropy Policies and League-Based Self-Play},
  author = {Mogharabin, Abtin and Sharifi, Kourosh},
  year   = {2026},
  note   = {Final Project Report for ML4350: Reinforcement Learning, University of Tübingen},
  url    = {https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent}
}
```
