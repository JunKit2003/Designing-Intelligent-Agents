import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random

# -------------------------
# Define the policy network
# -------------------------
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, num_phases):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        # Head for discrete phase selection:
        self.phase_head = nn.Linear(64, num_phases)
        # Head for continuous green duration adjustment:
        self.duration_mean = nn.Linear(64, 1)
        self.duration_log_std = nn.Parameter(torch.zeros(1))  # learnable log standard deviation

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        phase_logits = self.phase_head(x)
        duration_mean = self.duration_mean(x)
        duration_log_std = self.duration_log_std.expand_as(duration_mean)
        return phase_logits, duration_mean, duration_log_std

# -------------------------
# Define the value network
# -------------------------
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value_head(x)
        return value

# -------------------------
# PPO Agent for a Traffic Light
# -------------------------
class TrafficLightPPOAgent:
    def __init__(self, traffic_light_id, num_phases, state_dim=3, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=4):
        self.tl_id = traffic_light_id
        self.num_phases = num_phases
        self.state_dim = state_dim
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs

        self.policy_net = PolicyNetwork(state_dim, num_phases)
        self.value_net = ValueNetwork(state_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)

        # Trajectory storage
        self.states = []
        self.actions_phase = []
        self.actions_duration = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def get_state(self, traci):
        """
        Enhanced state representation.
          - total_queue: sum of waiting vehicles on all controlled lanes
          - avg_speed: average speed on controlled lanes (0 if no vehicles)
          - emergency_flag: 1 if any controlled lane has an emergency vehicle, 0 otherwise
        """
        lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        total_queue = 0
        speeds = []
        emergency_flag = 0
        for lane in lanes:
            n = traci.lane.getLastStepVehicleNumber(lane)
            total_queue += n
            speed = traci.lane.getLastStepMeanSpeed(lane)
            if speed is None or speed < 0:
                speed = 0
            speeds.append(speed)
            # Check for emergency vehicles:
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vid in vehicle_ids:
                if "emergency" in traci.vehicle.getTypeID(vid).lower():
                    emergency_flag = 1
        avg_speed = np.mean(speeds) if speeds else 0
        state = np.array([total_queue, avg_speed, emergency_flag], dtype=np.float32)
        return state

    def select_action(self, state):
        """
        Given a state, select an action:
          - Discrete: phase index
          - Continuous: green duration adjustment (e.g., in seconds)
        Returns the action tuple, its log probability, and the state value estimate.
        """
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # [1, state_dim]
        phase_logits, duration_mean, duration_log_std = self.policy_net(state_tensor)
        # Discrete action for phase
        phase_prob = F.softmax(phase_logits, dim=-1)
        phase_dist = torch.distributions.Categorical(phase_prob)
        phase_action = phase_dist.sample()
        phase_log_prob = phase_dist.log_prob(phase_action)
        # Continuous action for duration adjustment
        duration_std = torch.exp(duration_log_std)
        duration_dist = torch.distributions.Normal(duration_mean, duration_std)
        duration_action = duration_dist.sample()
        duration_log_prob = duration_dist.log_prob(duration_action).sum(dim=-1)
        # Combined log probability
        log_prob = phase_log_prob + duration_log_prob
        # State value estimate
        value = self.value_net(state_tensor)
        action = (phase_action.item(), duration_action.item())
        return action, log_prob.item(), value.item()

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions_phase.append(action[0])
        self.actions_duration.append(action[1])
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_returns_and_advantages(self, last_value, lam=0.95, normalize=True):
        """
        Compute discounted returns and advantages using Generalized Advantage Estimation (GAE).
        """
        returns = []
        advantages = []
        gae = 0
        values = self.values + [last_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * (1 - self.dones[step]) - values[step]
            gae = delta + self.gamma * lam * (1 - self.dones[step]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)
        if normalize:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def update(self, next_state, done):
        """
        Update PPO agent using the collected trajectory.
        """
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        last_value = self.value_net(next_state_tensor).item() if not done else 0

        returns, advantages = self.compute_returns_and_advantages(last_value)

        states_tensor = torch.tensor(self.states, dtype=torch.float32)
        actions_phase_tensor = torch.tensor(self.actions_phase, dtype=torch.long)
        actions_duration_tensor = torch.tensor(self.actions_duration, dtype=torch.float32).unsqueeze(1)
        old_log_probs_tensor = torch.tensor(self.log_probs, dtype=torch.float32).unsqueeze(1)

        for _ in range(self.update_epochs):
            phase_logits, duration_mean, duration_log_std = self.policy_net(states_tensor)
            phase_prob = F.softmax(phase_logits, dim=-1)
            phase_dist = torch.distributions.Categorical(phase_prob)
            new_phase_log_probs = phase_dist.log_prob(actions_phase_tensor).unsqueeze(1)

            duration_std = torch.exp(duration_log_std)
            duration_dist = torch.distributions.Normal(duration_mean, duration_std)
            new_duration_log_probs = duration_dist.log_prob(actions_duration_tensor).sum(dim=-1, keepdim=True)

            new_log_probs = new_phase_log_probs + new_duration_log_probs

            ratio = torch.exp(new_log_probs - old_log_probs_tensor)

            values = self.value_net(states_tensor)
            value_loss = F.mse_loss(values, returns.unsqueeze(1))

            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.unsqueeze(1)
            policy_loss = -torch.min(surr1, surr2).mean()

            loss = policy_loss + 0.5 * value_loss  # (Optionally, add an entropy bonus)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear stored trajectory
        self.states = []
        self.actions_phase = []
        self.actions_duration = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def decay_epsilon(self):
        # Not used in PPO (exploration is inherent to the stochastic policy).
        pass
