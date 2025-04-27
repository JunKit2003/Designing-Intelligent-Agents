# ppo_agent.py

import os
import math
import time
from collections import deque
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData
from torch_geometric.data.data import BaseData

# Import model and necessary components
import env
from model import TemporalGATTransformer, PPOWrapper


class PPOMemory:
    """
    Memory buffer for storing trajectories during PPO training
    """
    def __init__(self, buffer_size, batch_size):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
    def store(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.old_log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear(self):
        self.states = []
        self.actions = []
        self.old_log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        
    def __len__(self):
        return len(self.states)
    
    def get_batches(self):
        """Generate random batches from stored trajectories"""
        indices = torch.randperm(len(self.states))
        batch_starts = torch.arange(0, len(self.states), self.batch_size)
        
        batches = []
        for start_idx in batch_starts:
            end_idx = min(start_idx + self.batch_size, len(self.states))
            batch_indices = indices[start_idx:end_idx]
            
            batch_states = [self.states[i] for i in batch_indices]
            batch_actions = torch.stack([self.actions[i] for i in batch_indices])
            batch_log_probs = torch.stack([self.old_log_probs[i] for i in batch_indices])
            batch_values = torch.stack([self.values[i] for i in batch_indices])
            
            batches.append((batch_states, batch_actions, batch_log_probs, batch_values))
            
        return batches


class PPOAgent:
    """
    PPO Agent that integrates the TemporalGATTransformer model for traffic signal control
    """
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 8,
        transformer_layers: int = 2,
        history_length: int = 5,
        gat_layers: int = 2,
        clip_ratio: float = 0.2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048
    ):
        # Initialize model
        self.model = TemporalGATTransformer(
            in_features=state_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            transformer_layers=transformer_layers,
            history_length=history_length,
            gat_layers=gat_layers
        )
        
        # Create PPO wrapper for model
        self.policy = PPOWrapper(self.model)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        
        # PPO parameters
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        
        # Memory setup
        self.memory = PPOMemory(buffer_size, batch_size)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)
        
        # For tracking statistics
        self.training_step = 0
        self.episode_rewards = []
        self.cumulative_rewards = 0
        
    def compute_gae_returns(self, rewards, values, dones, next_value):
        """
        Compute generalized advantage estimation and returns
        
        Args:
            rewards: List of rewards
            values: List of predicted values
            dones: List of done flags
            next_value: Value estimate for next state
            
        Returns:
            advantages, returns
        """
        gae = 0
        advantages = []
        
        # Convert to tensors if needed
        if not isinstance(rewards[0], torch.Tensor):
            rewards = [torch.tensor(r).to(self.device) for r in rewards]
            
        if not isinstance(values[0], torch.Tensor):
            values = [torch.tensor(v).to(self.device) for v in values]
            
        if not isinstance(dones[0], torch.Tensor):
            dones = [torch.tensor(d, dtype=torch.float).to(self.device) for d in dones]
            
        if not isinstance(next_value, torch.Tensor):
            next_value = torch.tensor(next_value).to(self.device)
        
        # Reverse iteration for GAE calculation
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_val = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_val = values[t+1]
                
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            
        # Calculate returns (advantages + values)
        returns = [adv + val for adv, val in zip(advantages, values)]
        
        # Normalize advantages
        advantages_tensor = torch.stack(advantages)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
        
        return advantages_tensor, torch.stack(returns)
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """
        Update policy using PPO objective
        
        Args:
            states: List of states
            actions: Tensor of actions
            old_log_probs: Tensor of old log probabilities
            advantages: Tensor of advantages
            returns: Tensor of returns
            
        Returns:
            Dictionary of metrics
        """
        # Process batch of states
        states_batch = Batch.from_data_list(states)
        
        # Forward pass to get action distributions and values
        _, new_log_probs, entropy, values = self.policy(states_batch, actions)
        
        # PPO policy loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy loss
        entropy_loss = entropy.mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        # Gradient step
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # Track approximate KL divergence
        approx_kl = ((old_log_probs - new_log_probs).mean()).item()
        
        # Compile metrics
        metrics = {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'kl': approx_kl,
            'ratio': ratio.mean().item()
        }
        
        return metrics
    
    def collect_rollouts(self, env, n_steps):
        """
        Collect experience for training
        
        Args:
            env: Environment
            n_steps: Number of steps to collect
            
        Returns:
            True if completed successfully
        """
        # Reset memory
        self.memory.clear()
        
        # Get initial state
        state = env.get_observation()
        done = False
        step_counter = 0
        
        # Start collecting
        while step_counter < n_steps and not done:
            # Convert state to hetero data if needed
            if not isinstance(state, HeteroData):
                # Handle your specific state format here
                pass
                
            # Get action from policy
            with torch.no_grad():
                actions, log_probs, _, values = self.policy([state])
                
            # Convert actions to dictionary for environment
            action_dict = {}
            for i, tl_id in enumerate(env.tl_ids):
                action_dict[tl_id] = actions[i].item()
                
            # Take action in environment
            next_state, reward, done, _ = env.step(action_dict)
            
            # Track rewards for statistics
            self.cumulative_rewards += sum(reward) if isinstance(reward, list) else reward
            
            # Store transition
            self.memory.store(
                state,
                actions[0],  # Assuming single action tensor
                log_probs[0],
                values[0],
                torch.tensor(reward).to(self.device) if not isinstance(reward, torch.Tensor) else reward.to(self.device),
                torch.tensor(done).to(self.device) if not isinstance(done, torch.Tensor) else done.to(self.device)
            )
            
            # Move to next state
            state = next_state
            step_counter += 1
            
            # Record episode rewards if done
            if done:
                self.episode_rewards.append(self.cumulative_rewards)
                self.cumulative_rewards = 0
                
        return True
    
    def train(self, env, total_timesteps, log_interval=10, save_interval=1000, checkpoint_dir='checkpoints'):
        """
        Train the agent
        
        Args:
            env: Environment
            total_timesteps: Total training timesteps
            log_interval: How often to log training metrics
            save_interval: How often to save model checkpoints
            checkpoint_dir: Directory for saving checkpoints
        """
        # Create checkpoint directory
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Reset environment
        state = env.reset()
        
        # Track steps and iterations
        timesteps_so_far = 0
        update_iterations = 0
        
        # Training loop
        while timesteps_so_far < total_timesteps:
            # Collect experience
            self.collect_rollouts(env, self.memory.buffer_size)
            timesteps_so_far += len(self.memory)
            
            # Exit if we don't have enough data
            if len(self.memory) == 0:
                continue
                
            # Compute advantages and returns
            with torch.no_grad():
                # Get next state value
                next_value = self.policy.get_value([self.memory.states[-1]]) \
                             if not env.done else torch.zeros(1).to(self.device)
                             
            # Calculate GAE
            advantages, returns = self.compute_gae_returns(
                self.memory.rewards,
                self.memory.values,
                self.memory.dones,
                next_value
            )
            
            # Update policy multiple times
            total_metrics = {}
            
            for _ in range(self.update_epochs):
                # Generate random batches
                for batch_states, batch_actions, batch_log_probs, batch_values in self.memory.get_batches():
                    # Get batch indices for advantages and returns
                    batch_indices = [self.memory.states.index(state) for state in batch_states]
                    batch_advantages = advantages[batch_indices]
                    batch_returns = returns[batch_indices]
                    
                    # Update policy
                    metrics = self.update(
                        batch_states,
                        batch_actions,
                        batch_log_probs,
                        batch_advantages,
                        batch_returns
                    )
                    
                    # Track metrics
                    for k, v in metrics.items():
                        if k not in total_metrics:
                            total_metrics[k] = 0
                        total_metrics[k] += v
            
            # Average metrics
            for k in total_metrics:
                total_metrics[k] /= (self.update_epochs * len(self.memory.get_batches()))
                
            # Log metrics
            update_iterations += 1
            if update_iterations % log_interval == 0:
                print(f"Updates: {update_iterations}, Steps: {timesteps_so_far}")
                print(f"Policy Loss: {total_metrics['policy_loss']:.4f}, Value Loss: {total_metrics['value_loss']:.4f}")
                print(f"Entropy: {total_metrics['entropy']:.4f}, KL: {total_metrics['kl']:.4f}")
                print(f"Average Reward: {sum(self.episode_rewards[-10:]) / max(len(self.episode_rewards[-10:]), 1):.2f}")
                print("-" * 50)
                
            # Save checkpoint
            if update_iterations % save_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"ppo_checkpoint_{timesteps_so_far}.pt")
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'timesteps': timesteps_so_far,
                    'updates': update_iterations
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
    
    def act(self, state, deterministic=False):
        """
        Get action from policy
        
        Args:
            state: Environment state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Dictionary of actions for each traffic light
        """
        # Ensure model is in eval mode
        self.policy.eval()
        
        # Convert state if needed
        if not isinstance(state, HeteroData):
            # Handle conversion
            pass
            
        # Get action from policy
        with torch.no_grad():
            if deterministic:
                # Get logits and use argmax
                logits, _ = self.model(state)
                actions = torch.argmax(logits, dim=-1)
            else:
                # Sample from distribution
                actions, _, _, _ = self.policy([state])
                
        # Convert to dictionary of actions
        action_dict = {}
        for i, tl_id in enumerate(env.tl_ids):
            action_dict[tl_id] = actions[i].item()
            
        return action_dict
    
    def train_iteration(self):
        """
        Perform a training iteration using collected experiences.
        Returns: Dictionary of training metrics
        """
        # Exit if we don't have enough data
        if len(self.memory) == 0:
            return {"policy_loss": 0, "value_loss": 0, "entropy": 0, "kl": 0, "phase_switches": 0}
        
        # Compute advantages and returns
        with torch.no_grad():
            # Get next state value
            next_value = self.policy.get_value([self.memory.states[-1]]) \
                    if len(self.memory.dones) == 0 or not self.memory.dones[-1] \
                    else torch.zeros(1).to(self.device)
        
        # Calculate GAE
        advantages, returns = self.compute_gae_returns(
            self.memory.rewards,
            self.memory.values,
            self.memory.dones,
            next_value
        )
        
        # Update policy multiple times
        total_metrics = {}
        
        for _ in range(self.update_epochs):
            # Generate random batches
            for batch_states, batch_actions, batch_log_probs, batch_values in self.memory.get_batches():
                # Get batch indices for advantages and returns
                batch_indices = [self.memory.states.index(state) for state in batch_states]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Update policy
                metrics = self.update(
                    batch_states,
                    batch_actions,
                    batch_log_probs,
                    batch_advantages,
                    batch_returns
                )
                
                # Track metrics
                for k, v in metrics.items():
                    if k not in total_metrics:
                        total_metrics[k] = 0
                    total_metrics[k] += v
        
        # Average metrics
        n_updates = max(1, self.update_epochs * len(self.memory.get_batches()))
        for k in total_metrics:
            total_metrics[k] /= n_updates
        
        # Add phase switch tracking
        phase_switches = 0
        if len(self.memory.actions) > 1:
            for i in range(1, len(self.memory.actions)):
                if torch.any(self.memory.actions[i] != self.memory.actions[i-1]):
                    phase_switches += 1
        
        total_metrics['phase_switches'] = phase_switches
        
        # Clear memory after training
        self.memory.clear()
        
        return total_metrics

    
    def save(self, path):
        """Save model to path"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path):
        """Load model from path"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.policy.eval()
        
    def train_mode(self):
        """Set model to training mode"""
        self.policy.train()

