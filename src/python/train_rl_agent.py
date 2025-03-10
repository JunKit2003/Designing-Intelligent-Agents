import traci
import os
import sys
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from halo import Halo  # For a spinner during training
from rl_agent import TrafficLightPPOAgent  # Import the PPO agent
# (Ensure that PyTorch is installed)

# Ensure SUMO_HOME is set
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# Paths to SUMO scripts and files
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "..", "tools")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_FILE = os.path.join(OUTPUT_DIR, "sumo_errors.log")

SUMO_CONFIG = os.path.join(PROJECT_ROOT, "..", "config", "GeorgeTown.sumo.cfg")
NETWORK_FILE = os.path.join(PROJECT_ROOT, "..", "network", "GeorgeTown.net.xml")
TRIPS_FILE = os.path.join(PROJECT_ROOT, "..", "routes", "GeorgeTown.trips.xml")
ROUTES_FILE = os.path.join(PROJECT_ROOT, "..", "routes", "GeorgeTown.rou.xml")
RANDOM_TRIPS_SCRIPT = os.path.join(TOOLS_DIR, "randomTrips.py")  # Local copy

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run PPO-controlled SUMO traffic simulation.")
parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI visualization")
args = parser.parse_args()

SUMO_BINARY = "sumo-gui" if args.gui else "sumo"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Function to generate new randomized routes
def generate_random_routes():
    print("🔄 Generating new traffic routes...")
    if not os.path.exists(RANDOM_TRIPS_SCRIPT):
        sys.exit(f"Error: {RANDOM_TRIPS_SCRIPT} not found. Ensure 'randomTrips.py' is in the tools directory.")
    os.system(f'python "{RANDOM_TRIPS_SCRIPT}" -n "{NETWORK_FILE}" -o "{TRIPS_FILE}" --end 3600 --validate --fringe-factor 5')
    os.system(f'duarouter --net-file "{NETWORK_FILE}" --route-files "{TRIPS_FILE}" --output-file "{ROUTES_FILE}"')
    print("✅ New routes generated.")

# Function to save the best model (overwrite previous best)
def save_best_model(agents, avg_reward):
    model_path = os.path.join(MODEL_DIR, "best_model.npz")
    model_data = {tl: agent.policy_net.state_dict() for tl, agent in agents.items()}
    # Save using NumPy's savez; you might also use torch.save for a PyTorch checkpoint.
    np.savez(model_path, **model_data)
    print(f"💾 Best model updated: {model_path} (Avg Reward: {avg_reward:.2f})")

# Initialize reward tracking for graphing
episode_rewards = []
best_avg_reward = float('-inf')

# Training parameters
NUM_EPISODES = 1000
MAX_STEPS = 500

# Main training loop
for episode in range(NUM_EPISODES):
    print(f"\n🚦 Starting Episode {episode + 1}/{NUM_EPISODES}")
    generate_random_routes()  # Generate new traffic scenario
    
    # Launch SUMO with suppressed warnings/logs
    sumo_cmd = [
        SUMO_BINARY, "-c", SUMO_CONFIG, "--start",
        "--no-warnings", "--no-step-log", "--error-log", LOG_FILE
    ]
    traci.start(sumo_cmd)
    
    # Get traffic light IDs
    traffic_lights = traci.trafficlight.getIDList()
    
    # Create PPO agents for each traffic light.
    # (Here we create a separate PPO agent per intersection.
    # In practice, you might use weight sharing for scalability.)
    agents = {
        tl: TrafficLightPPOAgent(tl, len(traci.trafficlight.getAllProgramLogics(tl)[0].phases))
        for tl in traffic_lights
    }
    
    total_reward = 0
    
    # Start a spinner for training progress
    spinner = Halo(text=f"Training Episode {episode + 1}/{NUM_EPISODES}", spinner="dots")
    spinner.start()
    
    for step in range(MAX_STEPS):
        traci.simulationStep()
        
        # For each traffic light, get state, select action, apply it, and store transition.
        for tl, agent in agents.items():
            state = agent.get_state(traci)
            action, log_prob, value = agent.select_action(state)
            # Apply the action:
            # 1. Set the phase (discrete action)
            traci.trafficlight.setPhase(tl, action[0])
            # 2. Adjust the phase duration (continuous action)
            #    (Assume you have a mechanism in your simulation to adjust duration, e.g.,
            #     using traci.trafficlight.setPhaseDuration. Here we add the adjustment.)
            traci.trafficlight.setPhaseDuration(tl, max(5, action[1] + 30))  # base duration 30 seconds, adjusted by action
            
            traci.simulationStep()  # Observe effect
            
            # Compute reward using enhanced function:
            # reward = 0.1 * avg_speed - total_queue - 5*emergency_flag
            s = state  # state is [total_queue, avg_speed, emergency_flag]
            reward = 0.1 * s[1] - s[0] - 5 * s[2]
            total_reward += reward
            # Store transition (here, we assume 'done' is False for all steps until episode end)
            agent.store_transition(state, action, log_prob, reward, value, done=0)
    
    avg_reward = total_reward / MAX_STEPS
    episode_rewards.append(avg_reward)
    
    # For each agent, update using the collected trajectory.
    for tl, agent in agents.items():
        # Get next state (we use current state as the final state)
        next_state = agent.get_state(traci)
        agent.update(next_state, done=1)
    
    spinner.succeed(f"✅ Episode {episode + 1} Complete - Avg Reward: {avg_reward:.2f}")
    
    # Save best model if improved
    if avg_reward > best_avg_reward:
        best_avg_reward = avg_reward
        save_best_model(agents, avg_reward)
    
    traci.close()
    
    # Plot and save training progress graph (overwrite previous file)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker="o", linestyle="-", color="b", label="Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("PPO Training Progress")
    plt.legend()
    plt.grid()
    graph_path = os.path.join(OUTPUT_DIR, "training_progress.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"📊 Graph saved: {graph_path} (Overwritten every episode)")

print("🏁 Training Complete! Warnings & logs stored in", LOG_FILE)
