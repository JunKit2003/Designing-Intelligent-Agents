# dqn_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from noisy_dueling_model import NoisyDuelingNet
from utils import save_model, load_model

class NoisyDoubleDQNAgent:
    """
    A Double DQN Agent using a Noisy Dueling architecture for
    traffic signal control (single intersection or single environment).
    
    Now uses performance-based adaptive sigma for exploration:
      - Tracks recent episode rewards in a deque
      - If the latest reward is significantly lower than the recent average
        => increase sigma (re-exploration)
      - Otherwise => normal sigma decay
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-4,
        gamma=0.99,
        initial_sigma=0.5,       # Starting noise scale
        sigma_decay=0.995,       # Normal decay factor
        sigma_min=0.01,          # Minimum sigma
        target_update=10,
        device=None
    ):
        """
        Args:
            state_dim (int): Dimension of the state space.
            action_dim (int): Number of discrete actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            initial_sigma (float): Initial sigma for noisy layers.
            sigma_decay (float): Multiplicative decay factor per training update.
            sigma_min (float): Minimum sigma scale.
            target_update (int): Update target network every N steps.
            device: PyTorch device (defaults to CUDA if available).
        """
        # Device
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Policy and Target Networks
        self.policy_net = NoisyDuelingNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            initial_sigma=initial_sigma
        ).to(self.device)

        self.target_net = NoisyDuelingNet(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=128,
            initial_sigma=initial_sigma
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Hyperparameters
        self.gamma = gamma
        self.target_update = target_update
        self.action_dim = action_dim

        # Steps
        self.steps_done = 0

        # Adaptive Sigma Parameters
        self.sigma = initial_sigma
        self.sigma_decay = sigma_decay
        self.sigma_min = sigma_min
        self.sigma_max = 1.0             # We won't let sigma exceed 1.0
        self.sigma_increase = 1.2        # Factor to boost sigma if performance drops
        self.drop_threshold = 0.02       # 2% performance drop triggers exploration

        # Reward history for performance-based adaptation
        self.reward_history = deque(maxlen=10)  # last 10 episodes

    def select_action(self, state):
        """
        Select an action using the NoisyNet approach.
        We reset noise before each forward pass for fresh exploration.

        Args:
            state (np.array): Current environment state.

        Returns:
            int: Chosen action index.
        """
        with torch.no_grad():
            self.policy_net.reset_noise()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()
        return action   

    def update(self, replay_buffer, batch_size, episode_reward=None, episode_num=None):
        """
        Perform a training step using a batch from the replay buffer,
        and adapt sigma based on episode reward.

        Args:
            replay_buffer: ReplayBuffer with experiences.
            batch_size (int): Batch size for sampling.
            episode_reward (float): The final reward for the last episode
                                    (for performance-based sigma logic).

        Returns:
            float: Loss value for logging.
        """
        # Need enough samples to train
        if len(replay_buffer) < batch_size:
            return 0.0

        # Periodically update target net
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.steps_done += 1

        # -------------------------------
        # Adaptive Sigma Logic
        # -------------------------------
        if episode_num is not None and episode_num >= 450:
            if episode_reward is not None:
                self.reward_history.append(episode_reward)

            if len(self.reward_history) == self.reward_history.maxlen:
                recent_mean = sum(list(self.reward_history)[:-1]) / (self.reward_history.maxlen - 1)
                latest_reward = self.reward_history[-1]

                if latest_reward < recent_mean * (1.0 - self.drop_threshold):
                    self.sigma = min(self.sigma * self.sigma_increase, self.sigma_max)
                else:
                    self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        self.policy_net.adjust_sigma(self.sigma)
        self.target_net.adjust_sigma(self.sigma)

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Reset noise for forward pass
        self.policy_net.reset_noise()

        # Q-values for current actions
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Step 1: pick best actions in next_states using current (policy) net
        next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)

        # Double DQN Step 2: evaluate next_actions with target net
        with torch.no_grad():
            self.target_net.reset_noise()
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)

        # Target Q
        target_q_values = rewards + self.gamma * next_q_values * (~dones)

        # Loss
        loss = F.smooth_l1_loss(q_values, target_q_values.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save_models(self, path):
        """
        Save the policy and target networks to a directory.
        """
        save_model(self.policy_net, f"{path}/policy_net.pt")
        save_model(self.target_net, f"{path}/target_net.pt")

    def load_models(self, path):
        """
        Load the policy and target networks from a directory.
        """
        load_model(self.policy_net, f"{path}/policy_net.pt")
        load_model(self.target_net, f"{path}/target_net.pt")


class SharedNoisyDoubleDQNAgent:
    """
    A multi-intersection (shared) version of Noisy Double DQN
    for traffic signal control. Intersections are grouped by type,
    each group shares a single NoisyDoubleDQNAgent instance.
    """
    def __init__(
        self,
        state_dims,
        action_dims,
        intersection_groups,
        learning_rate=1e-4,
        gamma=0.99,
        initial_sigma=0.5,
        sigma_decay=0.995,
        sigma_min=0.01,
        target_update=10,
        device=None
    ):
        """
        Args:
            state_dims (dict): group_id -> state dimension
            action_dims (dict): group_id -> action dimension
            intersection_groups (dict): group_id -> list of traffic light IDs
            learning_rate (float): LR for each agent
            gamma (float): discount factor
            initial_sigma (float): initial sigma for each agent
            sigma_decay (float): normal decay factor for each agent
            sigma_min (float): floor for sigma
            target_update (int): freq of target net updates
            device: cpu or cuda
        """
        self.intersection_groups = intersection_groups
        self.agents = {}

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a NoisyDoubleDQNAgent for each group
        for group_id in intersection_groups:
            self.agents[group_id] = NoisyDoubleDQNAgent(
                state_dim=state_dims[group_id],
                action_dim=action_dims[group_id],
                learning_rate=learning_rate,
                gamma=gamma,
                initial_sigma=initial_sigma,
                sigma_decay=sigma_decay,
                sigma_min=sigma_min,
                target_update=target_update,
                device=device
            )

    def select_action(self, tl_id, state, group_id=None):
        """
        For a given traffic light (tl_id), pick an action from the
        agent that controls that traffic light's group.
        """
        if group_id is None:
            for g_id, tl_ids in self.intersection_groups.items():
                if tl_id in tl_ids:
                    group_id = g_id
                    break

        return self.agents[group_id].select_action(state)

    def update(self, group_id, replay_buffer, batch_size, episode_reward=None, episode_num=None):
        """
        Update the policy network for a specific group with a minibatch,
        optionally passing the final episode reward for performance-based sigma.
        """
        return self.agents[group_id].update(replay_buffer, batch_size, episode_reward, episode_num)

    def save_models(self, path):
        """
        Save all group models to given directory.
        """
        for group_id, agent in self.agents.items():
            save_model(agent.policy_net, f"{path}/policy_net_{group_id}.pt")
            save_model(agent.target_net, f"{path}/target_net_{group_id}.pt")

    def load_models(self, path):
        for group_id, agent in self.agents.items():
            load_model(agent.policy_net, 
                    f"{path}/policy_net_{group_id}.pt",
                    device=self.device)
            load_model(agent.target_net,
                    f"{path}/target_net_{group_id}.pt",
                    device=self.device)
