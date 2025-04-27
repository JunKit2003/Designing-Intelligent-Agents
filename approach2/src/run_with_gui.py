# run_with_gui.py

import os
import sys
import traci
import torch
import argparse

# Attempt to import tkinter
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# Adjust path if needed
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sumo_env import SumoEnvironment
from dqn_agent import SharedNoisyDoubleDQNAgent
from replay_buffer import ReplayBuffer
from utils import (
    set_seed,
    create_intersection_groups,
    get_intersection_group,
    create_state_action_dims
)


def run_fixed_time(config_file, max_steps=3600, seed=42):
    """
    Runs SUMO in GUI mode with built-in or default (fixed-time) traffic light program.
    """
    set_seed(seed)
    env = SumoEnvironment(
        config_file=config_file,
        use_gui=True,
        num_seconds=max_steps,
        delta_time=5
    )
    states = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1
        # We do not override traffic lights. Keep them on their current phase.
        actions = {}
        for tl_id in env.traffic_lights:
            actions[tl_id] = env.traffic_light_states[tl_id]['current_phase']

        next_states, rewards, dones, _ = env.step(actions)
        if len(rewards) > 0:
            step_reward = sum(rewards.values()) / len(rewards)
            total_reward += step_reward

        if any(dones.values()) or (step_count >= max_steps):
            done = True

        states = next_states

    env.close()
    print(f"\n🏁 Fixed-time run finished after {step_count} steps.")
    print(f"📈 Sum of average step reward: {total_reward:.2f}")


def run_trained_model(config_file, models_folder, max_steps=3600, seed=42):
    """
    Load a previously trained RL model from `models_folder`, then run with SUMO GUI.
    """
    set_seed(seed)
    env = SumoEnvironment(
        config_file=config_file,
        use_gui=True,
        num_seconds=max_steps,
        delta_time=5
    )

    intersection_groups = create_intersection_groups(env.intersection_types)
    print("Intersection Groups:")
    for group_id, tl_ids in intersection_groups.items():
        print(f"  {group_id} => {tl_ids}")

    state_dims, action_dims = create_state_action_dims(env)
    print("State Dims:", state_dims)
    print("Action Dims:", action_dims)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    agent = SharedNoisyDoubleDQNAgent(
        state_dims=state_dims,
        action_dims=action_dims,
        intersection_groups=intersection_groups,
        learning_rate=1e-4,
        gamma=0.99,
        initial_sigma=1.0,
        sigma_decay=0.995,
        sigma_min=0.01,
        target_update=10,
        device=device
    )

    agent.load_models(models_folder)
    print(f"✅ Loaded agent models from: {models_folder}")

    states = env.reset()
    done = False
    step_count = 0
    total_reward = 0.0

    while not done:
        step_count += 1
        actions = {}
        for tl_id in env.traffic_lights:
            group_id = get_intersection_group(tl_id, intersection_groups)
            action = agent.select_action(tl_id, states[tl_id], group_id)
            actions[tl_id] = action

        next_states, rewards, dones, _ = env.step(actions)
        step_reward = sum(rewards.values()) / len(rewards)
        total_reward += step_reward

        if any(dones.values()) or (step_count >= max_steps):
            done = True

        states = next_states

    env.close()
    print(f"\n🏁 Trained model run finished after {step_count} steps.")
    print(f"📈 Sum of average step reward: {total_reward:.2f}")


if __name__ == "__main__":
    default_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../run.sumocfg"))

    parser = argparse.ArgumentParser(description="Run SUMO in GUI with either fixed-time or a trained RL model.")
    parser.add_argument("--config_file", type=str, default=default_path, help="Path to the SUMO config file.")
    parser.add_argument("--max_steps", type=int, default=3600, help="Max steps (seconds) per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    print("Select Mode:\n  1) Fixed-time run (no RL)\n  2) Load a trained RL model")
    choice = input("Enter '1' or '2': ").strip()

    if choice == "1":
        print("✅ You chose FIXED-TIME mode.")
        run_fixed_time(
            config_file=args.config_file,
            max_steps=args.max_steps,
            seed=args.seed
        )

    elif choice == "2":
        print("✅ You chose TRAINED MODEL mode.")
        # Attempt to open a folder picker if tkinter is available
        model_folder = None

        if TKINTER_AVAILABLE:
            print("📂 Opening a folder chooser for your saved model files...")
            try:
                root = tk.Tk()
                root.withdraw()
                selected_folder = filedialog.askdirectory(title="Select Model Folder")
                if selected_folder:
                    model_folder = selected_folder
                else:
                    print("❌ No folder selected in the dialog.")
            except Exception as e:
                print(f"⚠️  Failed to open folder chooser: {e}")

        # If no folder chosen via GUI, ask user to type one
        while not model_folder:
            fallback = input("📝 Please type/paste the path to your model folder, or press Enter to cancel: ").strip()
            if fallback == "":
                print("❌ No folder provided. Exiting.")
                sys.exit(0)
            if os.path.isdir(fallback):
                model_folder = fallback
            else:
                print("❌ That path doesn't exist. Try again or press Enter to cancel.")

        run_trained_model(
            config_file=args.config_file,
            models_folder=model_folder,
            max_steps=args.max_steps,
            seed=args.seed
        )

    else:
        print("❌ Invalid choice. Please run again and pick '1' or '2'.")
