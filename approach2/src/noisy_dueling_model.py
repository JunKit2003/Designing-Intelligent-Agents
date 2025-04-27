# noisy_dueling_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.017, use_cuda=False):

        self.use_cuda = torch.cuda.is_available()

        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_cuda = use_cuda

        # Mean and sigma parameters for weights and biases
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))

        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.sigma_init = sigma_init

        # Initialize parameters and reset noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        if self.use_cuda:
            self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features).cuda())
            self.bias_epsilon.copy_(torch.randn(self.out_features).cuda())
        else:
            self.weight_epsilon.copy_(torch.randn(self.out_features, self.in_features))
            self.bias_epsilon.copy_(torch.randn(self.out_features))

    def forward(self, input):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(input, weight, bias)

    def set_sigma(self, sigma):
        """
        If you want to directly set the absolute sigma value
        (unused by default, but available if you prefer).
        """
        self.weight_sigma.data.fill_(sigma / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(sigma / np.sqrt(self.out_features))

    def get_noise_norm(self):
        """
        Compute norm of the noise parameters (optional debugging).
        """
        return self.weight_sigma.norm() ** 2 + self.bias_sigma.norm() ** 2

    def adjust_sigma(self, sigma_scale):
        """
        The agent calls this to scale the original sigma_init
        by some factor (sigma_scale). For example, if sigma_scale
        is decaying each step from 1.0 down to 0.01, this multiplies
        your initial sigma_init by that factor.
        """
        # fill with (sigma_init * sigma_scale / sqrt(...))
        self.weight_sigma.data.fill_(
            (self.sigma_init * sigma_scale) / np.sqrt(self.in_features)
        )
        self.bias_sigma.data.fill_(
            (self.sigma_init * sigma_scale) / np.sqrt(self.out_features)
        )


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
