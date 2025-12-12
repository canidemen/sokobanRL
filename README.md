# Sokoban RL with PPO

CS 175 Final Project - Training a Proximal Policy Optimization (PPO) agent to solve Sokoban puzzles.

## Quick Start (For Graders)

### Option 1: Run the Demo (Fastest - 2 minutes)

1. Open `src/sokoban_notebook.ipynb` in Jupyter Notebook or VS Code
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
│   ├── sokoban_notebook.ipynb    # Main notebook (START HERE)
│   ├── ppo_agent.py               # Standalone PPO implementation
│   └── test_checkpoint.py         # Evaluation script
├── checkpoints/                   # Saved model checkpoints
│   └── ppo_sokoban_ep3000.pth    # Pre-trained model (included)
├── puzzle_videos/                 # Generated visualization videos
├── logs/                          # Training logs
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

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
jupyter notebook src/sokoban_notebook.ipynb
```

### Google Colab Setup

```python
# Run this in a Colab cell
!git clone <repository-url> sokobanRL
%cd sokobanRL
!pip install -r requirements.txt

# Upload checkpoint file to /content/checkpoints/
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

- **Success Rate**: ~18-22% on Sokoban-small-v0
- **Average Reward**: ~-5 to +2 (after removing shaping bonuses)
- **Training Time**: ~8 hours on CPU
- **Final Episode**: 3000

Performance plateaus around episode 1500-2000, indicating the agent has learned effective strategies within the constraints of the small grid environment.

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

## Usage Examples

### Evaluate Pre-trained Agent

```python
# In notebook or Python script
from ppo_agent import PPOAgent
import gym
import gym_sokoban

env = gym.make('Sokoban-small-v0')
agent = PPOAgent(env)
agent.load('checkpoints/ppo_sokoban_ep3000.pth')

# Run 10 test episodes
for episode in range(10):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _ = agent.select_action(state)
        state, reward, done, info = env.step(action)
        total_reward += reward

    success = info.get('all_boxes_on_target', False)
    print(f"Episode {episode+1}: Reward={total_reward:.2f}, Success={success}")
```

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

### Generate Visualization Videos

```python
visualize_agent_pygame(
    checkpoint_path='checkpoints/ppo_sokoban_ep3000.pth',
    save_videos=True,
    output_dir='puzzle_videos',
    min_success_reward=10.0  # Only show high-quality successes
)
```

## Troubleshooting

### ImportError: No module named 'gym_sokoban'
```bash
pip install gym-sokoban
```

### Pygame window doesn't open
Make sure you're running locally, not on a headless server. For Google Colab, use the evaluation functions without visualization.

### FFMPEG errors when saving videos
```bash
pip install --upgrade imageio[ffmpeg]
```

### Checkpoint not found
Verify the checkpoint path:
- Local: `checkpoints/ppo_sokoban_ep3000.pth`
- Google Colab: `/content/checkpoints/ppo_sokoban_ep3000.pth`

## Performance Expectations

| Metric | Untrained Agent | Trained Agent (ep3000) |
|--------|----------------|------------------------|
| Success Rate | 0-1% | 18-22% |
| Avg Reward | -15 to -12 | -5 to +2 |
| Avg Steps | 150 (timeout) | 80-120 |
| Boxes on Target | 0-1 | 1-2 (complete) |

## Known Limitations

1. **Success Rate Plateau**: Performance plateaus around 20% due to:
   - Limited exploration in fixed-size grid
   - Difficulty generalizing across random puzzle configurations
   - Local optima in policy space

2. **Reward Farming**: Earlier versions learned to collect shaping rewards without solving puzzles. Fixed by:
   - Reducing reward magnitude (0.5→0.1, 2.0→0.5)
   - Adding timeout penalty (-3.0)
   - Making completion reward 20-100x more valuable

3. **Catastrophic Forgetting**: Some checkpoints show performance degradation after extended training. Mitigated by:
   - Entropy regularization
   - Value function clipping
   - Gradient clipping

## Future Improvements

- Curriculum learning: Start with 1-box puzzles, gradually increase difficulty
- Prioritized experience replay for rare successful trajectories
- Multi-environment training for better generalization
- Hierarchical RL: Learn sub-policies for box pushing, navigation, etc.
- Transfer learning to harder Sokoban variants (larger grids, more boxes)

## Citation

This project uses:
- **Proximal Policy Optimization (PPO)**: Schulman et al., 2017
- **gym-sokoban**: [mpSchrader/gym-sokoban](https://github.com/mpSchrader/gym-sokoban)
- **PyTorch**: [pytorch.org](https://pytorch.org/)

## License

Educational use for CS 175. See individual library licenses for dependencies.

## Contact

For questions or issues, please contact [your contact info here].
