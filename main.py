"""
Main entry point for Sokoban RL training and evaluation.
"""
import argparse
import torch
import os

from environment import SokobanWrapper
from models import SokobanCNN
from training import Trainer
from evaluation import Evaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sokoban RL Training and Evaluation')
    
    # Mode
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'both'],
                       help='Mode: train, eval, or both')
    
    # Training arguments
    parser.add_argument('--max_episodes', type=int, default=10000,
                       help='Maximum number of training episodes')
    parser.add_argument('--max_steps', type=int, default=200,
                       help='Maximum steps per episode')
    parser.add_argument('--update_frequency', type=int, default=128,
                       help='Steps before PPO update')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                       help='Number of PPO update epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for PPO updates')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--curriculum', action='store_true', default=True,
                       help='Enable curriculum learning')
    parser.add_argument('--no_curriculum', dest='curriculum', action='store_false',
                       help='Disable curriculum learning')
    
    # Evaluation arguments
    parser.add_argument('--eval_episodes', type=int, default=100,
                       help='Number of episodes for evaluation')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic policy for evaluation')
    
    # Checkpoint arguments
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to checkpoint file for evaluation or resuming')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='Directory to save/load checkpoints')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    
    return parser.parse_args()


def train(args):
    """Train the agent."""
    print("="*50)
    print("Training Mode")
    print("="*50)
    
    # Initialize trainer
    trainer = Trainer(
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        update_frequency=args.update_frequency,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        curriculum=args.curriculum,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir
    )
    
    # Load checkpoint if provided
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        trainer.load_checkpoint(args.checkpoint_path)
        print(f"Resuming training from checkpoint: {args.checkpoint_path}")
    
    # Train
    trainer.train()
    
    # Plot training curves
    training_history = {
        'episode_rewards': trainer.episode_rewards,
        'episode_lengths': trainer.episode_lengths,
        'success_rate_history': trainer.success_rate_history,
        'loss_history': trainer.loss_history
    }
    
    evaluator = Evaluator(trainer.model, trainer.env, args.results_dir)
    evaluator.plot_training_curves(training_history)
    
    return trainer


def evaluate(args):
    """Evaluate the agent."""
    print("="*50)
    print("Evaluation Mode")
    print("="*50)
    
    # Check checkpoint path
    if args.checkpoint_path is None:
        # Try to find latest checkpoint
        checkpoint_dir = args.checkpoint_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
            if checkpoints:
                args.checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Using checkpoint: {args.checkpoint_path}")
            else:
                raise ValueError("No checkpoint found. Please provide --checkpoint_path")
        else:
            raise ValueError("No checkpoint found. Please provide --checkpoint_path")
    
    # Initialize environment with improved generation parameters
    num_gen_steps = int(3.0 * (10 + 10))  # 60 steps for better room connectivity
    env = SokobanWrapper(
        dim_room=(10, 10),
        max_steps=args.max_steps,
        num_boxes=4,
        num_gen_steps=num_gen_steps
    )
    
    # Initialize model
    model = SokobanCNN(input_shape=(10, 10, 4), num_actions=9, hidden_size=256)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint: {args.checkpoint_path}")
    
    # Initialize evaluator
    evaluator = Evaluator(model, env, args.results_dir)
    
    # Evaluate
    metrics = evaluator.evaluate(
        num_episodes=args.eval_episodes,
        deterministic=args.deterministic
    )
    
    # Print and plot results
    evaluator.print_metrics(metrics)
    evaluator.plot_evaluation_results(metrics)
    
    return metrics


def main():
    """Main function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    import numpy as np
    np.random.seed(42)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        evaluate(args)
    elif args.mode == 'both':
        trainer = train(args)
        # Evaluate after training
        args.checkpoint_path = os.path.join(args.checkpoint_dir, 'final_checkpoint.pt')
        evaluate(args)


if __name__ == '__main__':
    main()
