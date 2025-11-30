"""
Evaluation suite and visualization tools.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from environment import SokobanWrapper
from models import SokobanCNN
from ppo import PPO


class Evaluator:
    """
    Evaluation suite for trained models.
    """
    
    def __init__(self, model, env, results_dir='./results'):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            env: Environment wrapper
            results_dir: Directory to save results
        """
        self.model = model
        self.env = env
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize PPO for action selection
        self.ppo = PPO(model)
    
    def evaluate(self, num_episodes=100, deterministic=False):
        """
        Evaluate model on multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic policy
            
        Returns:
            metrics: Dictionary with evaluation metrics
        """
        print(f"Evaluating model on {num_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        successes = []
        deadlocks = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            state = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done and episode_length < self.env.max_steps:
                # Get action
                action_mask = self.env.get_action_mask()
                action_mask_tensor = torch.BoolTensor(action_mask)
                
                action, _, _ = self.ppo.get_action(
                    torch.FloatTensor(state),
                    action_mask_tensor,
                    deterministic=deterministic
                )
                
                # Take step
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            # Check if solved
            room_state = self.env.env.room_state
            boxes_on_target = np.sum((room_state == 3).astype(int))
            total_boxes = self.env.num_boxes
            success = (boxes_on_target == total_boxes)
            
            # Check for deadlock (episode ended without success)
            deadlock = (not success) and (episode_length >= self.env.max_steps)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            successes.append(1.0 if success else 0.0)
            deadlocks.append(1.0 if deadlock else 0.0)
        
        # Compute metrics
        metrics = {
            'success_rate': np.mean(successes),
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'deadlock_rate': np.mean(deadlocks),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'successes': successes
        }
        
        return metrics
    
    def plot_training_curves(self, training_history, save_path=None):
        """
        Plot training curves.
        
        Args:
            training_history: Dictionary with training history
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        if 'episode_rewards' in training_history:
            rewards = training_history['episode_rewards']
            axes[0, 0].plot(rewards)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
            
            # Moving average
            if len(rewards) > 100:
                window = 100
                moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(rewards)), moving_avg, 
                               label='Moving Average (100)', color='red')
                axes[0, 0].legend()
        
        # Success rate
        if 'success_rate_history' in training_history:
            success_rates = training_history['success_rate_history']
            if len(success_rates) > 0:
                # Moving average
                window = 100
                if len(success_rates) > window:
                    moving_avg = np.convolve(success_rates, np.ones(window)/window, mode='valid')
                    axes[0, 1].plot(range(window-1, len(success_rates)), moving_avg)
                else:
                    axes[0, 1].plot(success_rates)
                axes[0, 1].set_title('Success Rate (Moving Average)')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Success Rate')
                axes[0, 1].grid(True)
                axes[0, 1].set_ylim([0, 1])
        
        # Episode lengths
        if 'episode_lengths' in training_history:
            lengths = training_history['episode_lengths']
            axes[1, 0].plot(lengths)
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Steps')
            axes[1, 0].grid(True)
        
        # Loss history
        if 'loss_history' in training_history:
            losses = training_history['loss_history']
            if len(losses) > 0:
                total_losses = [l['total_loss'] for l in losses]
                policy_losses = [l['policy_loss'] for l in losses]
                value_losses = [l['value_loss'] for l in losses]
                
                axes[1, 1].plot(total_losses, label='Total Loss')
                axes[1, 1].plot(policy_losses, label='Policy Loss')
                axes[1, 1].plot(value_losses, label='Value Loss')
                axes[1, 1].set_title('Training Losses')
                axes[1, 1].set_xlabel('Update')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved: {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_evaluation_results(self, metrics, save_path=None):
        """
        Plot evaluation results.
        
        Args:
            metrics: Evaluation metrics dictionary
            save_path: Path to save plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Reward distribution
        axes[0].hist(metrics['episode_rewards'], bins=20, edgecolor='black')
        axes[0].axvline(metrics['avg_reward'], color='red', linestyle='--', 
                       label=f"Mean: {metrics['avg_reward']:.2f}")
        axes[0].set_title('Episode Reward Distribution')
        axes[0].set_xlabel('Reward')
        axes[0].set_ylabel('Frequency')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Length distribution
        axes[1].hist(metrics['episode_lengths'], bins=20, edgecolor='black')
        axes[1].axvline(metrics['avg_length'], color='red', linestyle='--',
                       label=f"Mean: {metrics['avg_length']:.2f}")
        axes[1].set_title('Episode Length Distribution')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('Frequency')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Success metrics
        metrics_text = f"""
        Success Rate: {metrics['success_rate']:.2%}
        Deadlock Rate: {metrics['deadlock_rate']:.2%}
        Avg Reward: {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}
        Avg Length: {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}
        """
        axes[2].text(0.1, 0.5, metrics_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
        axes[2].set_title('Evaluation Metrics')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Evaluation results saved: {save_path}")
        else:
            plt.savefig(os.path.join(self.results_dir, 'evaluation_results.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def print_metrics(self, metrics):
        """Print evaluation metrics."""
        print("\n" + "="*50)
        print("Evaluation Metrics")
        print("="*50)
        print(f"Success Rate:        {metrics['success_rate']:.2%}")
        print(f"Deadlock Rate:      {metrics['deadlock_rate']:.2%}")
        print(f"Average Reward:     {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Average Length:     {metrics['avg_length']:.2f} ± {metrics['std_length']:.2f}")
        print("="*50 + "\n")
