"""
CNN architecture for policy and value networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SokobanCNN(nn.Module):
    """
    Convolutional Neural Network for Sokoban.
    Architecture: 3 conv layers + shared FC layers + separate policy/value heads.
    """

    # Large negative value for masking (instead of -inf to prevent NaN in log_softmax)
    LARGE_NEGATIVE = -1e8

    def __init__(self, input_shape=(10, 10, 4), num_actions=9, hidden_size=256):
        """
        Initialize CNN model.
        
        Args:
            input_shape: Shape of input state (height, width, channels)
            num_actions: Number of possible actions
            hidden_size: Size of hidden fully connected layers
        """
        super(SokobanCNN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Calculate flattened size after conv layers
        # Assuming input is (10, 10), after 3 conv layers with padding=1, size remains (10, 10)
        conv_output_size = 64 * input_shape[0] * input_shape[1]
        
        # Shared fully connected layers
        self.fc_shared1 = nn.Linear(conv_output_size, hidden_size)
        self.fc_shared2 = nn.Linear(hidden_size, hidden_size)
        
        # Policy head (actor)
        self.policy_head = nn.Linear(hidden_size, num_actions)
        
        # Value head (critic)
        self.value_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, action_mask=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor (batch, channels, height, width) or (batch, height, width, channels)
            action_mask: Optional mask for invalid actions (batch, num_actions)
            
        Returns:
            policy_logits: Logits for action probabilities (batch, num_actions)
            value: State value estimate (batch, 1)
        """
        # Handle input format: convert (batch, H, W, C) to (batch, C, H, W) if needed
        if x.dim() == 4 and x.shape[-1] == self.input_shape[2]:
            x = x.permute(0, 3, 1, 2).contiguous()  # (batch, H, W, C) -> (batch, C, H, W)
        
        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten (use reshape instead of view for better compatibility)
        x = x.reshape(x.size(0), -1)
        
        # Shared fully connected layers
        x = F.relu(self.fc_shared1(x))
        x = F.relu(self.fc_shared2(x))
        
        # Policy head
        policy_logits = self.policy_head(x)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Convert to tensor if numpy array
            if isinstance(action_mask, np.ndarray):
                action_mask = torch.BoolTensor(action_mask)
            # Ensure it's a tensor
            if not isinstance(action_mask, torch.Tensor):
                action_mask = torch.BoolTensor(action_mask)
            # Ensure it's on the same device as policy_logits
            if hasattr(action_mask, 'device') and action_mask.device != policy_logits.device:
                action_mask = action_mask.to(policy_logits.device)
            
            # Check if all actions are masked - if so, allow at least one action
            if action_mask.dim() == 2:
                # Batch case
                for i in range(action_mask.shape[0]):
                    if not action_mask[i].any():
                        # If all actions are masked, allow action 0 (no-op or first action)
                        action_mask[i, 0] = True
            elif not action_mask.any():
                # Single sample case - if all actions are masked, allow action 0
                action_mask[0] = True
            
            # Set logits for invalid actions to large negative value (not -inf to prevent NaN)
            policy_logits = policy_logits.masked_fill(~action_mask, self.LARGE_NEGATIVE)
        
        # Value head
        value = self.value_head(x)
        
        return policy_logits, value
    
    def get_action_and_value(self, x, action_mask=None):
        """
        Get action probabilities and state value.
        
        Args:
            x: Input state
            action_mask: Optional action mask
            
        Returns:
            action_probs: Action probabilities
            value: State value
            entropy: Policy entropy
        """
        policy_logits, value = self.forward(x, action_mask)
        
        # Get action probabilities
        action_probs = F.softmax(policy_logits, dim=-1)
        
        # Calculate entropy
        log_probs = F.log_softmax(policy_logits, dim=-1)
        entropy = -(action_probs * log_probs).sum(dim=-1, keepdim=True)
        
        return action_probs, value, entropy
