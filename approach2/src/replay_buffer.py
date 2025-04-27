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
