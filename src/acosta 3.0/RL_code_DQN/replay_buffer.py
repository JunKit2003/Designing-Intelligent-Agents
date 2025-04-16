# replay_buffer.py
import numpy as np
import random
from collections import deque


# try prioritized experience replay
class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling experiences
    """
    def __init__(self, capacity):
        """
        Initialize the replay buffer with a fixed capacity
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """
        Add an experience to the buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences from the buffer
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )
    
    def __len__(self):
        """Return the current size of the buffer"""
        return len(self.buffer)


# utils.py
import numpy as np
import torch
import os
import json
from collections import defaultdict

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_intersection_groups(intersection_types):
    """
    Group intersections by their type for model sharing
    
    Args:
        intersection_types: Dictionary mapping type signatures to lists of traffic light IDs
        
    Returns:
        Dictionary mapping group IDs to lists of traffic light IDs
    """
    return {f"group_{i}": tl_ids for i, tl_ids in enumerate(intersection_types.values())}

def get_intersection_group(tl_id, intersection_groups):
    """
    Get the group ID for a traffic light
    
    Args:
        tl_id: Traffic light ID
        intersection_groups: Dictionary mapping group IDs to lists of traffic light IDs
        
    Returns:
        Group ID for the traffic light
    """
    for group_id, tl_ids in intersection_groups.items():
        if tl_id in tl_ids:
            return group_id
    return None

def create_state_action_dims(env):
    """
    Create dictionaries of state and action dimensions for each intersection group
    
    Args:
        env: SumoEnvironment instance
        
    Returns:
        state_dims: Dictionary mapping group IDs to state dimensions
        action_dims: Dictionary mapping group IDs to action dimensions
    """
    # Create intersection groups
    intersection_groups = create_intersection_groups(env.intersection_types)
    
    # Get a representative traffic light for each group
    group_representatives = {}
    for group_id, tl_ids in intersection_groups.items():
        group_representatives[group_id] = tl_ids[0]
    
    # Get state and action dimensions for each group
    state_dims = {}
    action_dims = {}
    
    for group_id, tl_id in group_representatives.items():
        # Get state dimension from a sample state
        sample_state = env.get_state(tl_id)
        state_dims[group_id] = sample_state.shape[0]
        
        # Get action dimension from the number of phases
        action_dims[group_id] = len(env.phases[tl_id])
    
    return state_dims, action_dims

def save_model(model, path):
    """
    Save a PyTorch model
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load a PyTorch model
    
    Args:
        model: PyTorch model
        path: Path to the saved model
        
    Returns:
        Loaded model
    """
    model.load_state_dict(torch.load(path))
    return model

def save_metrics(metrics, path):
    """
    Save training metrics
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save the metrics
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def moving_average(values, window):
    """
    Calculate moving average
    
    Args:
        values: List of values
        window: Window size
        
    Returns:
        List of moving averages
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')
