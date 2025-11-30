"""
Training pipeline with curriculum learning.
"""
import torch
import numpy as np
from collections import deque
import os
from tqdm import tqdm

from environment import SokobanWrapper
from models import SokobanCNN
from ppo import PPO


class Trainer:
    """
    Training pipeline with curriculum learning and checkpointing.
    """
    
    def __init__(self,
                 max_episodes=10000,
                 max_steps=200,
                 update_frequency=128,
                 ppo_epochs=4,
                 batch_size=64,
                 lr=3e-4,
                 curriculum=True,
                 checkpoint_dir='./checkpoints',
                 results_dir='./results'):
        """
        Initialize trainer.
        
        Args:
            max_episodes: Maximum number of training episodes
            max_steps: Maximum steps per episode
            update_frequency: Steps before PPO update
            ppo_epochs: Number of PPO update epochs
            batch_size: Batch size for PPO updates
            lr: Learning rate
            curriculum: Whether to use curriculum learning
            checkpoint_dir: Directory to save checkpoints
            results_dir: Directory to save results
        """
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.update_frequency = update_frequency
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.curriculum = curriculum
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize environment - start with 2 boxes for curriculum learning
        self.base_dim_room = (10, 10)  # Keep 10x10 per project proposal
        self.base_num_boxes = 2  # Start with 2 boxes for easier learning
        self.target_num_boxes = 4  # Target is 4 boxes per proposal
        self.base_num_gen_steps = int(3.0 * sum(self.base_dim_room))  # 60 steps

        self.env = SokobanWrapper(
            dim_room=self.base_dim_room,
            max_steps=max_steps,
            num_boxes=self.base_num_boxes,
            num_gen_steps=self.base_num_gen_steps
        )

        # Initialize model (always 10x10 since room size doesn't change)
        self.model = SokobanCNN(input_shape=(10, 10, 4), num_actions=9, hidden_size=256)
        
        # Initialize PPO
        self.ppo = PPO(self.model, lr=lr)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate_history = []
        self.loss_history = []
        
        # Curriculum learning
        self.current_difficulty = 1  # Start with easier levels
        self.success_threshold = 0.7  # Increase difficulty when success rate > 70%
        
        # Experience buffer
        self.reset_buffer()
    
    def reset_buffer(self):
        """Reset experience buffer."""
        self.buffer = {
            'states': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'action_masks': [],
            'values': []
        }
    
    def collect_experience(self, num_steps):
        """
        Collect experience from environment.
        
        Args:
            num_steps: Number of steps to collect
        """
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(num_steps):
            # Get action mask
            action_mask = self.env.get_action_mask()
            action_mask_tensor = torch.BoolTensor(action_mask)
            
            # Get action from policy
            action, log_prob, value = self.ppo.get_action(
                torch.FloatTensor(state),
                action_mask_tensor
            )
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            self.buffer['states'].append(state)
            self.buffer['actions'].append(action)
            self.buffer['log_probs'].append(log_prob)
            self.buffer['rewards'].append(reward)
            self.buffer['dones'].append(done)
            self.buffer['action_masks'].append(action_mask)
            self.buffer['values'].append(value)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Store episode statistics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Check if puzzle was solved
                success = info.get('all_boxes_on_target', False) if isinstance(info, dict) else False
                self.success_rate_history.append(1.0 if success else 0.0)
                
                # Reset environment
                state = self.env.reset()
                episode_reward = 0
                episode_length = 0
        
        # Get value for next state (for GAE)
        if not done:
            action_mask = self.env.get_action_mask()
            action_mask_tensor = torch.BoolTensor(action_mask)
            _, _, next_value = self.ppo.get_action(
                torch.FloatTensor(state),
                action_mask_tensor
            )
        else:
            next_value = 0.0
        
        return next_value
    
    def update_policy(self, next_value=0):
        """
        Update policy using collected experience.
        
        Args:
            next_value: Value estimate for next state
        """
        # Convert buffers to arrays
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        old_log_probs = np.array(self.buffer['log_probs'])
        rewards = self.buffer['rewards']
        dones = np.array(self.buffer['dones'])
        action_masks = np.array(self.buffer['action_masks'])
        
        # Update for multiple epochs
        total_losses = []
        for epoch in range(self.ppo_epochs):
            # Shuffle indices
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            # Update in batches
            for i in range(0, len(states), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_rewards = [rewards[j] for j in batch_indices]
                batch_dones = dones[batch_indices]
                batch_action_masks = action_masks[batch_indices]
                
                # Get next value for last step in batch
                if i + self.batch_size >= len(states):
                    batch_next_value = next_value
                else:
                    batch_next_value = 0.0
                
                # Update
                loss_dict = self.ppo.update(
                    batch_states,
                    batch_actions,
                    batch_old_log_probs,
                    batch_rewards,
                    batch_dones,
                    batch_action_masks,
                    batch_next_value
                )
                total_losses.append(loss_dict)
        
        # Average losses
        avg_losses = {}
        for key in total_losses[0].keys():
            avg_losses[key] = np.mean([loss[key] for loss in total_losses])
        
        self.loss_history.append(avg_losses)
        return avg_losses
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        print(f"Max episodes: {self.max_episodes}")
        print(f"Update frequency: {self.update_frequency}")
        print(f"Curriculum learning: {self.curriculum}")
        
        episode_count = 0
        step_count = 0
        
        with tqdm(total=self.max_episodes, desc="Training") as pbar:
            while episode_count < self.max_episodes:
                # Collect experience
                next_value = self.collect_experience(self.update_frequency)
                step_count += self.update_frequency
                
                # Update policy
                if len(self.buffer['states']) > 0:
                    losses = self.update_policy(next_value)
                    
                    # Update progress bar
                    recent_success_rate = np.mean(self.success_rate_history[-100:]) if len(self.success_rate_history) > 0 else 0.0
                    recent_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) > 0 else 0.0
                    
                    pbar.set_postfix({
                        'episodes': episode_count,
                        'success_rate': f'{recent_success_rate:.2%}',
                        'avg_reward': f'{recent_reward:.2f}',
                        'loss': f'{losses["total_loss"]:.4f}'
                    })
                
                # Reset buffer
                self.reset_buffer()
                
                # Update episode count
                episode_count = len(self.episode_rewards)
                
                # Curriculum learning
                if self.curriculum and episode_count > 0 and episode_count % 100 == 0:
                    recent_success = np.mean(self.success_rate_history[-100:])
                    if recent_success > self.success_threshold and self.current_difficulty < 3:
                        self.current_difficulty += 1
                        print(f"\nIncreasing difficulty to level {self.current_difficulty}")
                        # Actually update environment parameters
                        self._update_curriculum_difficulty()
                
                # Save checkpoint
                if episode_count > 0 and episode_count % 1000 == 0:
                    self.save_checkpoint(f'checkpoint_ep{episode_count}.pt')
                
                pbar.update(min(100, self.max_episodes - episode_count))
        
        # Save final checkpoint
        self.save_checkpoint('final_checkpoint.pt')
        print("\nTraining completed!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.ppo.optimizer.state_dict(),
            'episode_count': len(self.episode_rewards),
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'success_rate_history': self.success_rate_history,
            'loss_history': self.loss_history,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ppo.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_lengths = checkpoint.get('episode_lengths', [])
        self.success_rate_history = checkpoint.get('success_rate_history', [])
        self.loss_history = checkpoint.get('loss_history', [])
        print(f"Checkpoint loaded: {checkpoint_path}")

    def _update_curriculum_difficulty(self):
        """Update environment parameters based on current difficulty level."""
        # Difficulty progression (keep 10x10 room, vary only num_boxes):
        # Level 1: 10x10, 2 boxes (easiest - starting point)
        # Level 2: 10x10, 3 boxes (intermediate)
        # Level 3: 10x10, 4 boxes (target per proposal)

        difficulty_configs = {
            1: {'num_boxes': 2},
            2: {'num_boxes': 3},
            3: {'num_boxes': 4}
        }

        config = difficulty_configs.get(self.current_difficulty, difficulty_configs[1])
        num_boxes = config['num_boxes']

        # Create new environment with updated parameters
        print(f"  Boxes: {num_boxes}, Gen steps: {self.base_num_gen_steps}")
        self.env = SokobanWrapper(
            dim_room=self.base_dim_room,
            max_steps=self.max_steps,
            num_boxes=num_boxes,
            num_gen_steps=self.base_num_gen_steps
        )
