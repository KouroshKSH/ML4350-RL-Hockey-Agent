# ShaBin: Laser Hockey Reinforcement Learning

This repository contains our contribution to the Reinforcement Learning Tournament of the University of Tübingen (Winter Term 2025/26) for the course **ML4350**, presented by [Prof. Dr. Georg Martius](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/distributed-intelligence/team/prof-dr-georg-martius/).

Our project focuses on training hockey agents to play the [Laser Hockey](https://github.com/antic11d/laser-hockey-env.git) game using off-policy algorithms and multi-agent training system.

![Laser Hockey Gameplay](assets/gameplay.gif "ShaBin Agent Gameplay")

## Tournament Results
Out of the competing teams, our solution was awarded **5th place**. 

| Rank | Algorithm | Author |
| :--- | :--- | :--- |
| 5 | [TQC](agent_1/tqc_agent.py) | Abtin Mogharabin |
| 5 | [SAC](agent_1/sac_agent.py) | Kourosh Sharifi |

*The official certificate for this achievement will be available [here]().*

## Core Features
- **Algorithms:** Implementation of Truncated Quantile Critics (TQC) and Soft Actor-Critic (SAC).
- **Reward Design:** Potential-based reward shaping, kinematic progress weights, and action $L_2$ regularization.
- **Self-Play Curriculum:** Opponent sampling via Prioritized Fictitious Self-Play (PFSP) and Upper Confidence Bound (UCB).
- **League Training:** A "lite" version of Policy-Space Response Oracles (PSRO) using Replicator Dynamics to ensure meta-strategy stability.

## Project Structure
Detailed information regarding the algorithms, ablation studies, and training results can be found in our **[Final Project Report](report.pdf)**.

If you wish to replicate our environment setup or training pipeline on a cluster (TCML), please refer to the file below:
👉 **[instructions.md](instructions.md)**

## Citation
If you find this work useful for your research or projects, please consider citing it:

```latex
@proceedings{shabin2026hockey,
  title  = {ShaBin: Competitive Laser Hockey via PSRO League Training and Truncated Quantile Critics},
  author = {Mogharabin, Abtin and Sharifi, Kourosh},
  year   = {2026},
  note   = {Final Project Report for ML4350: Reinforcement Learning, University of Tübingen},
  url    = {https://github.com/KouroshKSH/ML4350-RL-Hockey-Agent}
}
```
