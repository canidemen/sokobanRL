# Sokoban RL with PPO

CS 175 Final Project - Training a Proximal Policy Optimization (PPO) agent to solve Sokoban puzzles.

## Quick Start

### Option 1: Run the Demo (Fastest - 2 minutes)

1. Open `src/project.ipynb` in Jupyter Notebook or VS Code
2. Run cells 1-9 sequentially (setup, imports, class definitions)
3. **Run Cell 10 (DEMO)** - This will:
   - Load a pre-trained checkpoint (ep3000)
   - Evaluate the agent on 10 test episodes
   - Display success rate and performance metrics
   - Show a Pygame visualization of the agent solving puzzles

### Option 2: Full Evaluation (10 minutes)

1. Complete the Quick Start steps above
2. **Run Cell 11** - Comprehensive evaluation over 100 episodes
3. View detailed performance graphs and statistics

### Option 3: Train From Scratch (Several hours)

1. Complete setup steps (cells 1-9)
2. **Run Cell 12** - Trains a new agent from scratch
3. Monitor training progress in console output

## Project Structure

```
sokobanRL/
├── src/
│   ├── requirements.txt                # Python dependencies
│   └── project.ipynb                   # Main notebook (START HERE)
├── google_colabcheckpoints/            # Saved model checkpoints
│   └── sokoban-small-v0
│      └── ppo_sokoban_ep3000.pth          # Pre-trained model (included)
├── puzzle_videos/                      # Generated visualization videos
└── README.md                           # This file
```

## Installation

**Important**: Install Git LFS before cloning to prevent corruption of the pre-trained model file (`ppo_sokoban_ep3000.pth`):

```bash
git lfs install
```

### Local Setup

```bash
# 1. Clone/download the project
cd sokobanRL

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook src/project.ipynb
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- gym-sokoban
- pygame (for visualization)
- imageio[ffmpeg] (for video export)
- numpy, matplotlib, pandas

See `requirements.txt` for complete list.

## Environment Details

- **Environment**: `Sokoban-small-v0` (7x7 grid, 1-2 boxes)
- **Action Space**: Discrete(8) - up, down, left, right, push up/down/left/right
- **Observation Space**: RGB image (160, 160, 3)
- **Success Condition**: All boxes placed on target locations
- **Episode Length**: Max 150 steps

## Model Architecture

### Actor-Critic Network
- **Input**: 160x160x3 RGB image
- **Convolutional Layers**: 3 layers with Layer Normalization
  - Conv1: 32 filters, 3x3 kernel, stride 2
  - Conv2: 64 filters, 3x3 kernel, stride 2
  - Conv3: 64 filters, 3x3 kernel, stride 2
- **Actor Head**: Linear(conv_out, 128) → ReLU → Linear(128, 8)
- **Critic Head**: Linear(conv_out, 128) → ReLU → Linear(128, 1)
- **Initialization**: Orthogonal weight initialization

### PPO Hyperparameters
- Learning rate: 3e-4 (with 10-step warmup)
- Discount factor (γ): 0.99
- GAE lambda (λ): 0.95
- Clip epsilon: 0.2
- Value clip: 0.2
- Entropy coefficient: 0.05
- K epochs: 4
- Update frequency: Every 2048 timesteps
- Gradient clipping: 0.5

### Reward Shaping
To accelerate learning in this sparse reward environment, we add intermediate rewards:

| Event | Reward |
|-------|--------|
| Box moves closer to target | +0.1 per Manhattan distance unit |
| Box placed on target | +0.5 |
| Box moved away from target | -0.1 per unit |
| Puzzle solved | +10.0 (base environment) |
| Timeout without success | -3.0 |
| Each step | -0.1 (base environment) |

## Training Results

The included checkpoint (`ppo_sokoban_ep3000.pth`) was trained for 3000 episodes:

- **Success Rate**: ~10% on Sokoban-small-v0
- **Training Time**: ~8 hours on CPU
- **Final Episode**: 3000

Performance plateaus around episode 2000-2500, indicating the agent has learned effective strategies within the constraints of the small grid environment.

## Features

### 1. Reward Shaping
Provides intermediate feedback to guide learning in sparse reward environments.

### 2. Comprehensive Evaluation
- Per-episode success/failure tracking
- Statistical analysis (mean, median, quartiles)
- Success rate calculation
- Timeout detection
- Performance visualization

### 3. Pygame Visualization
- Real-time rendering of agent solving puzzles
- Records success and failure examples
- Exports to MP4 video format
- Configurable playback speed (default: 5 FPS)
- Text overlay showing episode info, steps, reward, and status

### 4. Checkpoint System
- Auto-saves every 100 episodes
- Resume training from any checkpoint
- Load for evaluation without retraining

### Train New Agent

```python
# See notebook Cell 12 for full training code
episode_rewards = train(
    env_name='Sokoban-small-v0',
    max_episodes=5000,
    max_timesteps=150,
    save_freq=100
)
```

### Checkpoint not found
Verify the checkpoint path: `google_colab_checkpoints/small-sokoban-v0/ppo_sokoban_ep3000.pth`

## Citation

This project uses:
- **Proximal Policy Optimization (PPO)**: Schulman et al., 2017
- **gym-sokoban**: [mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban)
- **PyTorch**: [pytorch.org](https://pytorch.org/)

## License

Educational use for CS 175. See individual library licenses for dependencies.
