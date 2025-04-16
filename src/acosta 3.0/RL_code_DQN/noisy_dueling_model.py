import torch
import torch.nn as nn
import torch.nn.functional as F

from NoisyLinear import NoisyLinear  # Factorized Noisy Linear layer


class NoisyDuelingNet(nn.Module):
    """
    A Dueling DQN network that uses NoisyLinear layers for exploration.
    This architecture is composed of:
      1) Two shared hidden layers (fc1, fc2)
      2) A separate "value" stream
      3) A separate "advantage" stream
      4) Recombination into final Q-values.

    Args:
        state_dim (int): Dimension of the state.
        action_dim (int): Number of discrete actions.
        hidden_dim (int): Number of units in the shared hidden layers.
        initial_sigma (float): Initial sigma (noise scale) for the NoisyLinear layers.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, initial_sigma=0.5):
        super(NoisyDuelingNet, self).__init__()

        # Shared feature layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value stream
        self.value_fc = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=initial_sigma),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 1, sigma_init=initial_sigma)
        )

        # Advantage stream
        self.advantage_fc = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, sigma_init=initial_sigma),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim, sigma_init=initial_sigma)

            
        )

    def forward(self, x):
        # Pass through the common hidden layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Compute value and advantage
        value = self.value_fc(x)           # shape: [batch_size, 1]
        advantage = self.advantage_fc(x)   # shape: [batch_size, action_dim]

        # Combine into Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_values = value + advantage - advantage_mean

        return q_values

    def reset_noise(self):
        """
        Reset the noise parameters in all NoisyLinear layers.
        This should be called before each forward pass during training
        to ensure fresh noise for exploration.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def adjust_sigma(self, sigma_scale):
        """
        Scale the sigma (noise level) in all NoisyLinear layers.
        Called by the agent to decay or change noise over time.
        """
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.adjust_sigma(sigma_scale)
