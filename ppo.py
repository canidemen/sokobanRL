"""
Proximal Policy Optimization (PPO) algorithm implementation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO:
    """
    Proximal Policy Optimization algorithm with:
    - Generalized Advantage Estimation (GAE)
    - Clipped surrogate objective
    - Value function learning
    """
    
    def __init__(self,
                 model,
                 lr=3e-4,
                 gamma=0.99,
                 gae_lambda=0.95,
                 clip_epsilon=0.2,
                 value_coef=0.5,
                 entropy_coef=0.02,
                 max_grad_norm=0.5):
        """
        Initialize PPO algorithm.
        
        Args:
            model: Policy and value network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip epsilon
            value_coef: Value loss coefficient
            entropy_coef: Entropy coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
    
    def compute_gae(self, rewards, values, dones, next_value=0):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Rewards for each step (T,)
            values: Value estimates for each step (T,)
            dones: Done flags for each step (T,)
            next_value: Value estimate for next state after last step
            
        Returns:
            advantages: Advantage estimates (T,)
            returns: Returns (discounted rewards) (T,)
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, rewards, dones, action_masks, next_value=0):
        """
        Update policy using PPO.
        
        Args:
            states: Batch of states (T, H, W, C)
            actions: Batch of actions (T,)
            old_log_probs: Old log probabilities (T,)
            rewards: Rewards (T,)
            dones: Done flags (T,)
            action_masks: Action masks (T, num_actions)
            next_value: Value estimate for next state
            
        Returns:
            loss_dict: Dictionary with loss components
        """
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Convert action_masks to tensor if they're numpy arrays
        if action_masks is not None:
            if isinstance(action_masks, np.ndarray):
                action_masks = torch.BoolTensor(action_masks)
            elif not isinstance(action_masks, torch.Tensor):
                action_masks = torch.BoolTensor(action_masks)
        
        # Get current policy and values
        policy_logits, values = self.model(states, action_masks)
        action_probs = torch.softmax(policy_logits, dim=-1)
        # Clamp probabilities to prevent log(0) and extreme values
        action_probs = torch.clamp(action_probs, min=1e-8, max=1.0)
        log_probs = torch.log_softmax(policy_logits, dim=-1)
        # Clamp log_probs to prevent extreme negative values
        log_probs = torch.clamp(log_probs, min=-20.0, max=0.0)
        
        # Get log probs for taken actions
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Get value estimates
        values = values.squeeze(1)
        values_np = values.detach().cpu().numpy()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values_np, dones, next_value)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # Normalize advantages with better numerical stability
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        # Only normalize if std is significant, otherwise just center
        if adv_std > 1e-5:
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)
        else:
            advantages = advantages - adv_mean
        
        # Compute policy loss (clipped surrogate objective)
        ratio = torch.exp(action_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_loss = 0.5 * (values - returns).pow(2).mean()
        
        # Compute entropy with NaN handling
        entropy_per_action = action_probs * log_probs
        # Replace any NaN values with 0 (contributes nothing to entropy)
        entropy_per_action = torch.where(
            torch.isnan(entropy_per_action),
            torch.zeros_like(entropy_per_action),
            entropy_per_action
        )
        entropy = -(entropy_per_action).sum(dim=-1).mean()
        # Clamp entropy to reasonable range
        entropy = torch.clamp(entropy, min=0.0, max=10.0)

        # Check for NaN in loss components and skip update if found
        if torch.isnan(policy_loss) or torch.isnan(value_loss) or torch.isnan(entropy):
            print(f"WARNING: NaN detected in loss - policy: {policy_loss.item():.4f}, "
                  f"value: {value_loss.item():.4f}, entropy: {entropy.item():.4f}")
            # Return loss dict with zeros to indicate failed update
            return {
                'total_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss': 0.0,
                'entropy': 0.0,
                'approx_kl': 0.0,
                'nan_detected': True
            }

        # Total loss
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Return loss components
        loss_dict = {
            'total_loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'approx_kl': (old_log_probs - action_log_probs).mean().item()
        }
        
        return loss_dict
    
    def get_action(self, state, action_mask=None, deterministic=False):
        """
        Get action from policy.
        
        Args:
            state: Current state (H, W, C) or (1, H, W, C)
            action_mask: Action mask (num_actions,) or (1, num_actions)
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        # Add batch dimension if needed
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if action_mask is not None and action_mask.dim() == 1:
            action_mask = action_mask.unsqueeze(0)
        
        # Get policy and value
        with torch.no_grad():
            policy_logits, value = self.model(state, action_mask)
            action_probs = torch.softmax(policy_logits, dim=-1)
            
            # Check for NaN values (can happen if all actions were masked)
            if torch.isnan(action_probs).any():
                # If NaN, create uniform distribution over valid actions
                if action_mask is not None:
                    # Use action mask to create uniform distribution
                    if isinstance(action_mask, torch.Tensor):
                        valid_mask = action_mask
                    else:
                        valid_mask = torch.ones_like(action_probs, dtype=torch.bool)
                    action_probs = valid_mask.float()
                    action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
                else:
                    # Uniform over all actions
                    action_probs = torch.ones_like(action_probs) / action_probs.shape[-1]
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1).item()
            else:
                # Sample from distribution
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()
            
            # Calculate log prob (handle NaN case)
            log_probs = torch.log_softmax(policy_logits, dim=-1)
            if torch.isnan(log_probs).any():
                # If log_probs has NaN, use log of action_probs instead
                log_prob = torch.log(action_probs[0, action] + 1e-8).item()
            else:
                log_prob = log_probs[0, action].item()
            
            value = value.item()
        
        return action, log_prob, value
