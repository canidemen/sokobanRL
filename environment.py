"""
Environment wrapper for Sokoban with preprocessing, reward shaping, and action masking.
"""
import gym
import gym_sokoban
from gym_sokoban.envs.sokoban_env import SokobanEnv
import numpy as np
import torch


class SokobanWrapper:
    """
    Wrapper for gym-sokoban environment that provides:
    - State preprocessing (10x10x4 tensor: walls, boxes, goals, player)
    - Action masking (filters invalid actions)
    - Reward shaping (intermediate rewards based on box-goal distances)
    - Deadlock detection
    """
    
    def __init__(self, dim_room=(10, 10), max_steps=200, num_boxes=4, num_gen_steps=None):
        """
        Initialize Sokoban environment wrapper.

        Args:
            dim_room: Room dimensions (height, width)
            max_steps: Maximum steps per episode
            num_boxes: Number of boxes in the puzzle
            num_gen_steps: Number of generation steps. If None, uses 3.0 * sum(dim_room)
        """
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes

        # Calculate num_gen_steps if not provided (3.0x instead of default 1.7x for better connectivity)
        if num_gen_steps is None:
            num_gen_steps = int(3.0 * (dim_room[0] + dim_room[1]))
        self.num_gen_steps = num_gen_steps

        # Create environment directly (gym.make doesn't pass kwargs properly)
        self.env = SokobanEnv(
            dim_room=dim_room,
            max_steps=max_steps,
            num_boxes=num_boxes,
            num_gen_steps=num_gen_steps
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right, 4=no-op, etc.
        self.action_deltas = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        
    def reset(self, max_retries=10):
        """
        Reset environment and return preprocessed state.
        Retries if generation fails.

        Args:
            max_retries: Maximum number of reset attempts

        Returns:
            state: Preprocessed state tensor
        """
        for attempt in range(max_retries):
            try:
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                state = self._preprocess_state(obs)
                return state
            except (RuntimeError, RuntimeWarning) as e:
                if attempt == max_retries - 1:
                    # On final attempt, use fallback: temporarily reduce boxes
                    print(f"WARNING: Failed to generate room after {max_retries} attempts. "
                          f"Using {self.num_boxes - 1} boxes as fallback.")
                    old_boxes = self.env.num_boxes
                    self.env.num_boxes = max(1, old_boxes - 1)
                    try:
                        obs = self.env.reset()
                    finally:
                        self.env.num_boxes = old_boxes  # Restore
                    if isinstance(obs, tuple):
                        obs = obs[0]
                    state = self._preprocess_state(obs)
                    return state
                else:
                    # Retry
                    continue

        # Should never reach here due to fallback, but just in case
        raise RuntimeError(f"Failed to reset environment after {max_retries} attempts")
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            state: Preprocessed state
            reward: Shaped reward
            done: Whether episode is done
            info: Additional information
        """
        obs, reward, done, info = self.env.step(action)
        if isinstance(obs, tuple):
            obs = obs[0]
        
        # Preprocess state
        state = self._preprocess_state(obs)
        
        # Shape reward
        shaped_reward = self._shape_reward(reward, obs, done)
        
        # Check for deadlock
        if not done:
            if self._is_deadlock(obs):
                done = True
                shaped_reward -= 5.0  # Penalty for deadlock
        
        return state, shaped_reward, done, info
    
    def _preprocess_state(self, obs):
        """
        Convert Sokoban state to 10x10x4 tensor.
        Channels: [walls, boxes, goals, player]
        
        Args:
            obs: Raw observation from environment
            
        Returns:
            state: Preprocessed state tensor (10, 10, 4)
        """
        # Get room state and fixed elements
        room_state = self.env.room_state.copy()
        room_fixed = self.env.room_fixed.copy()
        
        height, width = self.dim_room
        
        # Initialize state tensor
        state = np.zeros((height, width, 4), dtype=np.float32)
        
        # Channel 0: Walls (from room_fixed: 0=wall, 1=empty, 2=goal)
        state[:, :, 0] = (room_fixed == 0).astype(np.float32)
        
        # Channel 1: Boxes (from room_state: 4=box, 3=box_on_goal)
        state[:, :, 1] = ((room_state == 4) | (room_state == 3)).astype(np.float32)
        
        # Channel 2: Goals (from room_fixed: 2=goal)
        state[:, :, 2] = (room_fixed == 2).astype(np.float32)
        
        # Channel 3: Player (from room_state: 5=player)
        state[:, :, 3] = (room_state == 5).astype(np.float32)
        
        return state
    
    def _shape_reward(self, reward, obs, done):
        """
        Shape the reward with intermediate signals.

        Args:
            reward: Original reward from environment
            obs: Observation
            done: Whether episode is done

        Returns:
            shaped_reward: Shaped reward value
        """
        shaped_reward = reward

        if not done:
            # Add intermediate reward based on box-goal distances
            room_state = self.env.room_state
            room_fixed = self.env.room_fixed

            # Find boxes and goals
            boxes = np.argwhere((room_state == 4) | (room_state == 3))
            goals = np.argwhere(room_fixed == 2)

            if len(boxes) > 0 and len(goals) > 0:
                # Calculate minimum distance from each box to nearest goal
                min_distances = []
                for box in boxes:
                    distances = [np.abs(box[0] - goal[0]) + np.abs(box[1] - goal[1])
                               for goal in goals]
                    min_distances.append(min(distances))

                # Reward for boxes being closer to goals
                avg_distance = np.mean(min_distances)
                distance_reward = -0.01 * avg_distance  # Small negative reward for distance
                shaped_reward += distance_reward

        return shaped_reward
    
    def _is_deadlock(self, obs):
        """
        Simple deadlock detection: check if any box is stuck against walls.
        This is a simplified version - full deadlock detection is more complex.

        Args:
            obs: Observation

        Returns:
            is_deadlock: Whether state appears to be in deadlock
        """
        room_state = self.env.room_state
        room_fixed = self.env.room_fixed

        # Find boxes
        boxes = np.argwhere((room_state == 4) | (room_state == 3))

        for box in boxes:
            row, col = box
            # Check if box is in corner (simplified deadlock check)
            if room_fixed[row, col] != 2:  # Not on goal
                # Check if surrounded by walls on two adjacent sides
                walls_adjacent = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = row + dr, col + dc
                    if (0 <= nr < self.dim_room[0] and 0 <= nc < self.dim_room[1] and
                        room_fixed[nr, nc] == 0):  # Wall
                        walls_adjacent += 1

                # If box is in corner (2 adjacent walls), it might be deadlocked
                if walls_adjacent >= 2:
                    return True

        return False
    
    def get_action_mask(self):
        """
        Get action mask to filter invalid actions (moving into walls).
        
        Returns:
            action_mask: Boolean array of valid actions
        """
        room_state = self.env.room_state
        room_fixed = self.env.room_fixed
        
        # Find player position
        player_pos = np.argwhere(room_state == 5)
        if len(player_pos) == 0:
            # If player not found, allow all actions
            return np.ones(self.action_space.n, dtype=bool)
        
        player_row, player_col = player_pos[0]
        
        # Check each action
        action_mask = np.zeros(self.action_space.n, dtype=bool)
        
        for action in range(4):  # Only check movement actions
            if action in self.action_deltas:
                dr, dc = self.action_deltas[action]
                new_row, new_col = player_row + dr, player_col + dc
                
                # Check if new position is valid (not a wall)
                if (0 <= new_row < self.dim_room[0] and 
                    0 <= new_col < self.dim_room[1] and
                    room_fixed[new_row, new_col] != 0):  # Not a wall
                    action_mask[action] = True
                else:
                    action_mask[action] = False
        
        # Allow no-op and other actions
        for action in range(4, self.action_space.n):
            action_mask[action] = True
        
        # Safety check: ensure at least one action is valid
        if not action_mask.any():
            # If all actions are masked, allow action 0 (usually no-op or first action)
            action_mask[0] = True
        
        return action_mask
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode=mode)
    
    def close(self):
        """Close the environment."""
        self.env.close()
