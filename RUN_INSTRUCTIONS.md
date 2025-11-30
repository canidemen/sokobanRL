# How to Run the Sokoban RL Project

## Prerequisites
- Python 3.10 installed
- Virtual environment activated (`.venv` with Python 3.10)

## Step 1: Activate Virtual Environment

### Windows (PowerShell):
```powershell
cd "c:\Users\canid\OneDrive\Masaüstü\Python\CS_175\sokobanRL"
.\.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt):
```cmd
cd "c:\Users\canid\OneDrive\Masaüstü\Python\CS_175\sokobanRL"
.venv\Scripts\activate.bat
```

## Step 2: Install Dependencies

```bash
cd sokobanRL
pip install -r requirements.txt
```

This will install:
- gym>=0.21.0
- gym-sokoban>=0.0.5
- numpy>=1.20.0
- torch>=1.9.0
- matplotlib>=3.3.0
- tqdm>=4.60.0
- imageio>=2.9.0
- pillow>=8.0.0

## Step 3: Run the Project

### Option A: Training Only

Train the agent from scratch:
```bash
python main.py --mode train --max_episodes 10000
```

Train with custom parameters:
```bash
python main.py --mode train --max_episodes 5000 --max_steps 200 --lr 3e-4 --update_frequency 128
```

### Option B: Evaluation Only

Evaluate a trained model:
```bash
python main.py --mode eval --checkpoint_path ./checkpoints/final_checkpoint.pt --eval_episodes 100
```

### Option C: Training + Evaluation

Train and then automatically evaluate:
```bash
python main.py --mode both --max_episodes 10000 --eval_episodes 100
```

## Command Line Arguments

### Training Arguments:
- `--max_episodes`: Number of training episodes (default: 10000)
- `--max_steps`: Maximum steps per episode (default: 200)
- `--update_frequency`: Steps before PPO update (default: 128)
- `--ppo_epochs`: Number of PPO update epochs (default: 4)
- `--batch_size`: Batch size for PPO updates (default: 64)
- `--lr`: Learning rate (default: 3e-4)
- `--curriculum`: Enable curriculum learning (default: True)
- `--no_curriculum`: Disable curriculum learning

### Evaluation Arguments:
- `--eval_episodes`: Number of episodes for evaluation (default: 100)
- `--deterministic`: Use deterministic policy for evaluation

### Checkpoint Arguments:
- `--checkpoint_path`: Path to checkpoint file for evaluation or resuming
- `--checkpoint_dir`: Directory to save/load checkpoints (default: ./checkpoints)
- `--results_dir`: Directory to save results (default: ./results)

## Example Commands

### Quick test (small training run):
```bash
python main.py --mode train --max_episodes 100 --update_frequency 64
```

### Resume training from checkpoint:
```bash
python main.py --mode train --max_episodes 10000 --checkpoint_path ./checkpoints/checkpoint_ep1000.pt
```

### Evaluate with deterministic policy:
```bash
python main.py --mode eval --checkpoint_path ./checkpoints/final_checkpoint.pt --eval_episodes 50 --deterministic
```

## Output Files

After running, you'll find:
- **Checkpoints**: `./checkpoints/` - Model checkpoints saved during training
- **Results**: `./results/` - Training curves and evaluation plots
  - `training_curves.png` - Training progress visualization
  - `evaluation_results.png` - Evaluation metrics visualization

## Troubleshooting

### If you get import errors:
Make sure you're in the `sokobanRL` directory and the virtual environment is activated:
```bash
cd sokobanRL
# Activate venv (see Step 1)
pip install -r requirements.txt
```

### If gym-sokoban installation fails:
Try installing directly:
```bash
pip install gym-sokoban
```

### If you get CUDA/GPU errors:
The code will automatically use CPU if CUDA is not available. To force CPU:
```python
# In main.py, add before model initialization:
import torch
torch.set_default_tensor_type('torch.FloatTensor')
```

## Quick Start (Minimal Example)

```bash
# 1. Activate venv
.\.venv\Scripts\Activate.ps1

# 2. Navigate to project
cd sokobanRL

# 3. Install dependencies (if not already installed)
pip install -r requirements.txt

# 4. Run training
python main.py --mode train --max_episodes 1000
```

This will train for 1000 episodes and save checkpoints in `./checkpoints/`.
