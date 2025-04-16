# utils.py
import random
import numpy as np
import torch
import os
import json
from collections import defaultdict

torch.set_num_threads(24)

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


def save_model(model, path):
    """
    Save a PyTorch model
    
    Args:
        model: PyTorch model
        path: Path to save the model
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model, path, device=None):
    if device is not None:
        model.load_state_dict(
            torch.load(path, map_location=device)
        )
    else:
        model.load_state_dict(torch.load(path))
    model.eval()


