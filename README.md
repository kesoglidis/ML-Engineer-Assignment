# ML Engineer Assignment

A reinforcement learning project built using **Stable-Baselines3 (SB3)** and **Gymnasium**, featuring a custom **GridWorld environment** (`GridEnv`).  
The agent must navigate toward a target while avoiding obstacles and visiting **bonus positions** before reaching the final goal.

---

## Environment Setup

### Requirements

Ensure you have **Python 3.9+** installed, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure
```bash
GridEnv/
│
├── expanded_env.py          # Expanded environment with bonus positions agent must reach before reaching final target
├── train.py                 # Training script
├── evaluate.py              # Evaluation script
├── design.pdf               # Written design analysis
├── README.md                # This file
└── trained_models/          # Directory for trained models
```

## Run the training script:
```bash
python train.py```
```

The script will train an on-policy A2C agent and an off-policy DQN agent until convergence on three 6x6 grids with 5 obstacles and 0,2,5 bonus positions respectively.
While logging progress, which includes average reward and success rate. Finally, saving the converged model in the trained_models/ directory.

## Run the evaluation script:
```bash
python evaluation.py```
```
Run the evaluation script to test a trained model over 100 episodes

## Evaluation Metrics:

| Metric                        | Description                                      |
| ----------------------------- | ------------------------------------------------ |
| **Average Steps per Episode** | Mean number of steps before termination          |
| **Average Total Reward**      | Mean cumulative reward per episode               |
| **Success Rate**              | % of episodes where the agent reached the target |
| **Episode Statistics**        | Number of episode termination causes (e.g. Hit Wall/Obstacle, Reached Target) |

