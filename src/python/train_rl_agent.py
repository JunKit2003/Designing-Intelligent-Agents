#!/usr/bin/env python
import os
import sys
import time
import argparse
import multiprocessing
import matplotlib.pyplot as plt
from halo import Halo  # For a spinner during training
import traci
import torch
import numpy as np

# Import your PPO agent
from rl_agent import TrafficLightPPOAgent

# Ensure SUMO_HOME is set
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

# -------------------------
# Global Paths & Configurations
# -------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS_DIR = os.path.join(PROJECT_ROOT, "..", "tools")
# ROUTES_DIR: where generated trips and routes files will be stored
ROUTES_DIR = os.path.join(PROJECT_ROOT, "..", "routes")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "output")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_FILE = os.path.join(OUTPUT_DIR, "sumo_errors.log")

SUMO_CONFIG = os.path.join(PROJECT_ROOT, "..", "config", "GeorgeTown.sumo.cfg")
NETWORK_FILE = os.path.join(PROJECT_ROOT, "..", "network", "GeorgeTown.net.xml")
# Use our local copy of randomTrips.py from the tools folder:
RANDOM_TRIPS_SCRIPT = os.path.join(TOOLS_DIR, "randomTrips.py")

# -------------------------
# Parse Command-Line Arguments
# -------------------------
parser = argparse.ArgumentParser(
    description="Run PPO-controlled SUMO traffic simulation in parallel for different traffic levels."
)
parser.add_argument("--gui", action="store_true", help="Run SUMO with GUI visualization")
args = parser.parse_args()

SUMO_BINARY = "sumo-gui" if args.gui else "sumo"

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Function: Generate Random Routes for a Given Traffic Level
# -------------------------
def generate_random_routes(traffic_level):
    """
    Generate trips and routes for the specified traffic level.
    Different insertion rates are used for low, medium, and rush-hour traffic.
    The generated files are stored in the routes folder.
    """
    print(f"🔄 Generating {traffic_level} traffic routes...")

    if not os.path.exists(RANDOM_TRIPS_SCRIPT):
        sys.exit(f"Error: {RANDOM_TRIPS_SCRIPT} not found. Ensure 'randomTrips.py' is in the tools directory.")

    # Build output file names inside the routes folder:
    trips_file = os.path.join(ROUTES_DIR, f"GeorgeTown_{traffic_level}.trips.xml")
    routes_file = os.path.join(ROUTES_DIR, f"GeorgeTown_{traffic_level}.rou.xml")

    if traffic_level == "low":
        insertion_rate = 200  # Vehicles per hour
    elif traffic_level == "medium":
        insertion_rate = 600  # Vehicles per hour
    elif traffic_level == "rush":
        insertion_rate = 1800  # Vehicles per hour
    else:
        sys.exit("Invalid traffic level provided.")

    # Call randomTrips.py with both -o and -r options so that it generates both files.
    os.system(
        f'python "{RANDOM_TRIPS_SCRIPT}" -n "{NETWORK_FILE}" -o "{trips_file}" -r "{routes_file}" '
        f'--end 3600 --insertion-rate {insertion_rate} --validate --fringe-factor 5'
    )
    # Optionally, wait a short time to allow file operations to complete.
    time.sleep(2)
    print(f"✅ {traffic_level} traffic routes generated in {routes_file}.")
    return routes_file

# -------------------------
# Function: Save Best Model for a Given Traffic Level
# -------------------------
def save_best_model(agents, avg_reward, traffic_level):
    """
    Save the best model (policy network parameters) for a given traffic level.
    The model is saved in MODEL_DIR with the traffic level in the filename.
    """
    model_path = os.path.join(MODEL_DIR, f"{traffic_level}_traffic_model.pth")
    model_data = {tl: agent.policy_net.state_dict() for tl, agent in agents.items()}
    torch.save(model_data, model_path)
    print(f"💾 Best model updated for {traffic_level} traffic: {model_path} (Avg Reward: {avg_reward:.2f})")

# -------------------------
# Function: Train PPO Agents for a Given Traffic Level
# -------------------------
def train_agent_for_traffic_level(traffic_level):
    """
    Train PPO agents (one per traffic light) for the specified traffic level.
    Generates routes, launches SUMO with those routes, runs the training loop,
    saves the best model, and plots a training progress graph.
    """
    print(f"\n🚦 Starting training for {traffic_level} traffic...")
    routes_file = generate_random_routes(traffic_level)

    # Build SUMO command with the generated routes file
    sumo_cmd = [
        SUMO_BINARY, "-c", SUMO_CONFIG, "--start",
        "--route-files", routes_file,
        "--no-warnings", "--no-step-log", "--error-log", LOG_FILE
    ]
    traci.start(sumo_cmd)

    # Retrieve all traffic light IDs from the simulation
    traffic_lights = traci.trafficlight.getIDList()

    # Create PPO agents (one per intersection)
    agents = {
        tl: TrafficLightPPOAgent(tl, len(traci.trafficlight.getAllProgramLogics(tl)[0].phases))
        for tl in traffic_lights
    }

    # Training parameters
    NUM_EPISODES = 10000  # Adjust as needed
    MAX_STEPS = 500

    episode_rewards = []
    best_avg_reward = float('-inf')

    # Main training loop
    for episode in range(NUM_EPISODES):
        print(f"\n[{traffic_level.upper()}] 🚦 Starting Episode {episode + 1}/{NUM_EPISODES}")
        total_reward = 0

        # Start a spinner for training progress
        spinner = Halo(text=f"[{traffic_level.upper()}] Training Episode {episode + 1}/{NUM_EPISODES}", spinner="dots")
        spinner.start()

        for step in range(MAX_STEPS):
            traci.simulationStep()
            for tl, agent in agents.items():
                state = agent.get_state(traci)
                action, log_prob, value = agent.select_action(state)
                # Apply the discrete action: set phase
                traci.trafficlight.setPhase(tl, action[0])
                # Apply the continuous action: adjust phase duration (base duration 30 seconds)
                traci.trafficlight.setPhaseDuration(tl, max(5, action[1] + 30))
                traci.simulationStep()  # Advance simulation to observe effect

                # Compute reward:
                # For state = [total_queue, avg_speed, emergency_flag]
                # reward = 0.1 * avg_speed - total_queue - 5 * emergency_flag
                reward = 0.1 * state[1] - state[0] - 5 * state[2]
                total_reward += reward

                # Store transition (done=0 until episode end)
                agent.store_transition(state, action, log_prob, reward, value, done=0)

        avg_reward = total_reward / MAX_STEPS
        episode_rewards.append(avg_reward)

        # Update each agent using the collected trajectory
        for tl, agent in agents.items():
            next_state = agent.get_state(traci)
            agent.update(next_state, done=1)

        spinner.succeed(f"[{traffic_level.upper()}] ✅ Episode {episode + 1} Complete - Avg Reward: {avg_reward:.2f}")

        # Save best model if improved
        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            save_best_model(agents, avg_reward, traffic_level)

    traci.close()

    # Plot and save training progress graph (overwrite previous file for this traffic level)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker="o", linestyle="-", color="b", label="Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title(f"PPO Training Progress ({traffic_level.upper()} Traffic)")
    plt.legend()
    plt.grid()
    graph_path = os.path.join(OUTPUT_DIR, f"training_progress_{traffic_level}.png")
    plt.savefig(graph_path)
    plt.close()
    print(f"📊 Graph saved for {traffic_level} traffic: {graph_path} (Overwritten each run)")

    print(f"🏁 Training Complete for {traffic_level} traffic! Warnings & logs stored in {LOG_FILE}")

# -------------------------
# Main: Run Training in Parallel for Different Traffic Levels
# -------------------------
def main():
    # Define traffic levels for which to train separate agents
    traffic_levels = ["low", "medium", "rush"]

    processes = []
    for level in traffic_levels:
        p = multiprocessing.Process(target=train_agent_for_traffic_level, args=(level,))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    print("🏁 All traffic level agents have been trained successfully!")

if __name__ == "__main__":
    main()
