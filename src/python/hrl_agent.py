#!/usr/bin/env python
"""
hrl_agent.py

This script trains a high‐level reinforcement learning (HRL) agent that
performs per-intersection model selection (low, medium, rush) during a SUMO simulation.
It directs all SUMO warnings/errors to "hrl_sumo_errors.log" and uses a Halo spinner
to indicate that training is running. The HRL agent (a small PPO network) is trained
to decide, for each intersection, which lower-level model to use.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import traci
import traci.constants as tc
from halo import Halo

# -------------------------
# Global Paths & Configurations
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "..", "output", "models")
LOG_FILE = os.path.join(PROJECT_ROOT, "..", "output", "hrl_sumo_errors.log")
SUMO_CONFIG = os.path.join(PROJECT_ROOT, "..", "config", "GeorgeTown.sumo.cfg")
NETWORK_FILE = os.path.join(PROJECT_ROOT, "..", "network", "GeorgeTown.net.xml")

# Lower-level model checkpoint paths
LOW_MODEL_PATH = os.path.join(MODEL_DIR, "low_traffic_model.pth")
MEDIUM_MODEL_PATH = os.path.join(MODEL_DIR, "medium_traffic_model.pth")
RUSH_MODEL_PATH = os.path.join(MODEL_DIR, "rush_traffic_model.pth")

# -------------------------
# High-Level PPO Network (for per-intersection decisions)
# -------------------------
class HighLevelAgent(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32, num_actions=3):
        """
        input_dim: dimension of the state (e.g., [total_queue, avg_speed, emergency_flag])
        num_actions: 0 = low, 1 = medium, 2 = rush
        """
        super(HighLevelAgent, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

def select_high_level_action(agent, state):
    """
    Given a state (numpy array), compute logits and sample an action.
    Returns the action (int) and its log probability.
    """
    state_tensor = torch.tensor(state, dtype=torch.float32)
    logits = agent(state_tensor)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

# -------------------------
# Utility: Adjust loaded state_dict if dimensions don’t match (for lower-level agents)
# -------------------------
def adjust_state_dict(model, state_dict):
    model_dict = model.state_dict()
    new_state_dict = {}
    for key in model_dict.keys():
        if key in state_dict:
            loaded_param = state_dict[key]
            if loaded_param.shape != model_dict[key].shape:
                if "phase_head" in key:
                    if loaded_param.shape[0] > model_dict[key].shape[0]:
                        new_state_dict[key] = loaded_param[:model_dict[key].shape[0]]
                        print(f"Adjusted '{key}' from shape {loaded_param.shape} to {model_dict[key].shape}")
                    elif loaded_param.shape[0] < model_dict[key].shape[0]:
                        extra = model_dict[key][loaded_param.shape[0]:]
                        new_state_dict[key] = torch.cat([loaded_param, extra], dim=0)
                        print(f"Extended '{key}' from shape {loaded_param.shape} to {model_dict[key].shape}")
                    else:
                        new_state_dict[key] = loaded_param
                else:
                    new_state_dict[key] = loaded_param
            else:
                new_state_dict[key] = loaded_param
        else:
            new_state_dict[key] = model_dict[key]
    return new_state_dict

# -------------------------
# Lower-level PPO Agent (imported from your rl_agent.py)
# -------------------------
from rl_agent import TrafficLightPPOAgent

def load_checkpoint_for_agent(agent, checkpoint):
    """
    Loads a checkpoint into agent.policy_net.
    The checkpoint can be a unified state_dict or a dictionary mapping traffic light IDs.
    """
    if isinstance(checkpoint, dict) and all(isinstance(v, dict) for v in checkpoint.values()):
        # Per-intersection dictionary: if agent.tl_id is available, use that one
        if agent.tl_id in checkpoint:
            state_dict = checkpoint[agent.tl_id]
        else:
            state_dict = list(checkpoint.values())[0]
    else:
        state_dict = checkpoint
    try:
        agent.policy_net.load_state_dict(state_dict)
    except RuntimeError as e:
        state_dict = adjust_state_dict(agent.policy_net, state_dict)
        agent.policy_net.load_state_dict(state_dict)

# -------------------------
# Helper function to get per-intersection state
# -------------------------
def get_intersection_state(tl):
    lanes = traci.trafficlight.getControlledLanes(tl)
    total_queue = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in lanes)
    speeds = [traci.lane.getLastStepMeanSpeed(lane) or 0 for lane in lanes]
    avg_speed = np.mean(speeds) if speeds else 0
    emergency = 0
    for lane in lanes:
        for vid in traci.lane.getLastStepVehicleIDs(lane):
            if "emergency" in traci.vehicle.getTypeID(vid).lower():
                emergency = 1
                break
    return np.array([total_queue, avg_speed, emergency], dtype=np.float32)

# -------------------------
# Helper: Compute simple returns
# -------------------------
def compute_returns(rewards, gamma):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

# -------------------------
# High-Level Training Function
# -------------------------
def train_hrl_agent(num_episodes, gui=False):
    SUMO_BINARY = "sumo-gui" if gui else "sumo"
    # Create the high-level agent (PPO) for per-intersection decisions.
    high_level_agent = HighLevelAgent()
    hl_optimizer = optim.Adam(high_level_agent.parameters(), lr=1e-3)
    
    # Trajectory buffers for high-level agent (each decision is per intersection)
    hl_states, hl_actions, hl_log_probs, hl_rewards = [], [], [], []
    
    for ep in range(num_episodes):
        print(f"HRL Episode {ep+1}/{num_episodes}")
        spinner = Halo(text=f"HRL Episode {ep+1}/{num_episodes}", spinner="dots")
        spinner.start()
        
        # Start SUMO simulation (warnings/errors go to LOG_FILE)
        traci.start([SUMO_BINARY, "-c", SUMO_CONFIG, "--start", "--no-step-log", "--error-log", LOG_FILE])
        time.sleep(2)  # allow simulation to load
        
        traffic_lights = traci.trafficlight.getIDList()
        
        # High-level decision: For each intersection, get its state and choose a traffic model.
        per_int_states = {}
        per_int_actions = {}
        per_int_log_probs = {}
        for tl in traffic_lights:
            state = get_intersection_state(tl)
            per_int_states[tl] = state
            action, log_prob = select_high_level_action(high_level_agent, state)
            per_int_actions[tl] = action
            per_int_log_probs[tl] = log_prob
        
        # Map action (0=low, 1=medium, 2=rush) to corresponding model path.
        model_map = {0: LOW_MODEL_PATH, 1: MEDIUM_MODEL_PATH, 2: RUSH_MODEL_PATH}
        lower_agents = {}
        for tl in traffic_lights:
            num_phases = len(traci.trafficlight.getAllProgramLogics(tl)[0].phases)
            lower_agent = TrafficLightPPOAgent(tl, num_phases, state_dim=3)
            model_path = model_map[ per_int_actions[tl] ]
            if not os.path.exists(model_path):
                sys.exit(f"Model file for selected traffic level not found: {model_path}")
            loaded_state = torch.load(model_path, map_location=torch.device("cpu"))
            load_checkpoint_for_agent(lower_agent, loaded_state)
            lower_agents[tl] = lower_agent
        
        # Run simulation using the selected lower-level models for a fixed number of steps.
        sim_steps = 500
        for step in range(sim_steps):
            traci.simulationStep()
            for tl, agent in lower_agents.items():
                state = agent.get_state(traci)
                action, _, _ = agent.select_action(state)
                traci.trafficlight.setPhase(tl, action[0])
                traci.trafficlight.setPhaseDuration(tl, max(5, action[1] + 30))
            time.sleep(0.05)
        
        # After simulation, compute reward (here, negative total queue length aggregated).
        sim_reward = 0
        for tl in traffic_lights:
            state = get_intersection_state(tl)
            sim_reward += -state[0]
        avg_sim_reward = sim_reward / len(traffic_lights)
        
        # Store high-level trajectory for each intersection.
        for tl in traffic_lights:
            hl_states.append(per_int_states[tl])
            hl_actions.append(per_int_actions[tl])
            hl_log_probs.append(per_int_log_probs[tl])
            hl_rewards.append(avg_sim_reward)
        
        traci.close()
        spinner.succeed(f"HRL Episode {ep+1} complete - Avg Reward: {avg_sim_reward:.2f}")
        print(f"Aggregated avg queue: {np.mean([s[0] for s in hl_states]):.2f} -> Model choice distribution: {np.bincount(np.array(hl_actions))}")
        
        # High-level PPO update:
        returns = compute_returns(hl_rewards, gamma=0.99)
        states_tensor = torch.tensor(np.array(hl_states), dtype=torch.float32)
        actions_tensor = torch.tensor(hl_actions, dtype=torch.long)
        old_log_probs_tensor = torch.stack(hl_log_probs).detach()  # detach to avoid backprop through stored graph
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        advantages = returns_tensor - returns_tensor.mean()
        advantages = advantages.detach()  # detach advantages
        
        # Perform multiple epochs of PPO updates.
        for _ in range(4):
            logits = high_level_agent(states_tensor)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions_tensor)
            ratio = torch.exp(new_log_probs - old_log_probs_tensor)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
            loss = -torch.min(surr1, surr2).mean()
            hl_optimizer.zero_grad()
            loss.backward()
            hl_optimizer.step()
        
        # Clear high-level buffers after each episode.
        hl_states.clear()
        hl_actions.clear()
        hl_log_probs.clear()
        hl_rewards.clear()
        
        # Save high-level model checkpoint after each episode.
        checkpoint_path = os.path.join(MODEL_DIR, "hrl_model.pth")
        torch.save(high_level_agent.state_dict(), checkpoint_path)
        print(f"High-level HRL model saved to {checkpoint_path}.")
        time.sleep(1)
    
    print("HRL training complete.")

# -------------------------
# Main entry point
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Train HRL agent for per-intersection traffic model selection.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of HRL training episodes")
    parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI visualization")
    args = parser.parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_hrl_agent(num_episodes=args.episodes, gui=args.gui)

if __name__ == "__main__":
    main()
