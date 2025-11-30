# Sokoban RL

Reinforcement Learning agent for solving Sokoban puzzles using Proximal Policy Optimization (PPO) with CNN-based policy and value networks.

## Project Overview

This project implements a deep reinforcement learning agent that learns to solve Sokoban puzzles from grid-based state representations. The agent uses:
- **CNN Architecture**: Convolutional neural networks to process 10x10 grid states
- **PPO Algorithm**: Proximal Policy Optimization for stable policy learning
- **Reward Shaping**: Intermediate rewards based on box-goal distances
- **Action Masking**: Filters invalid actions to improve learning efficiency
- **Curriculum Learning**: Gradually increases difficulty as agent improves

## Team Members

- Hao Ding (14668568, haod8@uci.edu) - Environment integration, preprocessing, reward shaping, curriculum learning
- Can Idemen (53204603, cidemen@uci.edu) - CNN architecture, PPO algorithm implementation
- Carlos Arias (10081608, ariasc3@uci.edu) - Training pipeline, evaluation suite, visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sokobanRL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install gym-sokoban:
```bash
pip install gym-sokoban
```

## Project Structure

```
sokobanRL/
├── main.py              # Main entry point for training/evaluation
├── environment.py       # Environment wrapper with preprocessing and reward shaping
├── models.py            # CNN architecture for policy and value networks
├── ppo.py               # PPO algorithm implementation
├── training.py          # Training pipeline with curriculum learning
├── evaluation.py        # Evaluation suite and visualization tools
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Training

Train the agent from scratch:
```bash
python sokobanRL/main.py --mode train --max_episodes 10000
```

Training options:
- `--max_episodes`: Number of training episodes (default: 10000)
- `--max_steps`: Maximum steps per episode (default: 200)
- `--lr`: Learning rate (default: 3e-4)
- `--curriculum`: Enable curriculum learning (default: True)
- `--checkpoint_dir`: Directory to save checkpoints (default: ./checkpoints)
- `--results_dir`: Directory to save results (default: ./results)

### Evaluation

Evaluate a trained model:
```bash
python sokobanRL/main.py --mode eval --checkpoint_path ./checkpoints/final_checkpoint.pt --eval_episodes 100
```

### Training and Evaluation

Run both training and evaluation:
```bash
python sokobanRL/main.py --mode both --max_episodes 10000 --eval_episodes 100
```

## Features

### Environment Wrapper (`environment.py`)
- **State Preprocessing**: Converts Sokoban state to 10x10x4 tensor (walls, boxes, goals, player)
- **Action Masking**: Filters invalid actions (moving into walls)
- **Reward Shaping**: Provides intermediate rewards based on box-goal distances
- **Deadlock Detection**: Identifies and penalizes unsolvable states

### CNN Model (`models.py`)
- **Architecture**: 3 convolutional layers + shared fully connected layers
- **Dual Heads**: Separate policy (actor) and value (critic) heads
- **Action Masking Support**: Handles invalid action filtering

### PPO Algorithm (`ppo.py`)
- **GAE**: Generalized Advantage Estimation for stable learning
- **Clipped Surrogate Objective**: Prevents large policy updates
- **Value Function Learning**: Separate value network for baseline reduction

### Training Pipeline (`training.py`)
- **Curriculum Learning**: Starts with easier levels, progresses as agent improves
- **Episode Management**: Collects and batches experience for PPO updates
- **Checkpointing**: Saves model checkpoints periodically

### Evaluation Suite (`evaluation.py`)
- **Metrics**: Success rate, average steps, deadlock rate
- **Visualization**: Training curves and evaluation results plots
- **Statistics**: Comprehensive performance metrics

## Hyperparameters

Default hyperparameters (can be adjusted via command line):
- Learning rate: 3e-4
- Discount factor (gamma): 0.99
- GAE lambda: 0.95
- PPO clip epsilon: 0.2
- Value loss coefficient: 0.5
- Entropy coefficient: 0.01
- Update frequency: 128 steps
- PPO epochs: 4
- Batch size: 64

## Dataset

The project uses the **Boxoban Levels Dataset** from DeepMind:
- Training: ~900,000 levels
- Validation: ~100,000 levels
- Test: 1,000 levels

Dataset: https://github.com/google-deepmind/boxoban-levels

## Results

The agent is evaluated on:
- **Success Rate**: Percentage of puzzles solved
- **Average Steps**: Mean number of steps to solve puzzles
- **Deadlock Rate**: Percentage of episodes ending in deadlock

Training curves and evaluation results are saved in the `results/` directory.

## References

- Shoham, A., & Elidan, G. (2021). Forward-Backward Reinforcement Learning for Sokoban.
- Boxoban Levels Dataset: https://github.com/google-deepmind/boxoban-levels
- gym-sokoban: https://github.com/mpSchrader/gym-sokoban

## License

This project is for educational purposes as part of CS 175 at UC Irvine.
