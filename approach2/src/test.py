import os
import sys

from tqdm import tqdm # Import sys
import traci
import torch
import argparse
import time
import numpy as np
import csv
from datetime import datetime

# Attempt to import tkinter
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False

# --- Assuming these imports work because test.py is in RL_code_DQN ---
from sumo_env import SumoEnvironment
from dqn_agent import SharedNoisyDoubleDQNAgent
# ReplayBuffer might not be needed for testing, but keep if used indirectly
from replay_buffer import ReplayBuffer
from utils import (
    set_seed,
    create_intersection_groups,
    get_intersection_group,
    create_state_action_dims
)
# ---------------------------------------------------------------------

# --- Scenario Configuration ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to get the base directory containing both scenario folders
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

scenario_configs = {
    'acosta': {
        'dir': os.path.join(base_dir, 'acosta'),
        'bus_weight': 2.5,
        'metric_subdir': 'metric', # Subdirectory for metrics
        'model_subdir': 'output_acosta/model' # Default model location relative to scenario dir
    },
    'pasubio': {
        'dir': os.path.join(base_dir, 'pasubio'),
        'bus_weight': 5.0,
        'metric_subdir': 'metric', # Subdirectory for metrics
        'model_subdir': 'output_pasubio/model' # Default model location relative to scenario dir
    }
}
# ----------------------------

def select_scenario():
    """Prompts the user to select a scenario."""
    while True:
        print("\nSelect Scenario:")
        print("  1) Acosta")
        print("  2) Pasubio")
        choice = input("Enter '1' or '2': ").strip()
        if choice == '1':
            return 'acosta'
        elif choice == '2':
            return 'pasubio'
        else:
            print("Invalid choice. Please enter '1' or '2'.")


class MetricsCollector:
    """Collects and processes traffic metrics for different vehicle types."""

    # --- Modified __init__ to accept output directory ---
    def __init__(self, metric_output_dir, log_filename_prefix="metrics"):
        self.metrics = {
            'bus': {'waiting_times': [], 'speeds': [], 'queue_lengths': []},
            'other': {'waiting_times': [], 'speeds': [], 'queue_lengths': []},
            'all': {'waiting_times': [], 'speeds': [], 'queue_lengths': []}
        }
        self.step_metrics = {
            'bus': {'waiting_time': 0, 'speed': 0, 'count': 0},
            'other': {'waiting_time': 0, 'speed': 0, 'count': 0},
            'all': {'waiting_time': 0, 'speed': 0, 'count': 0}
        }
        self.queue_lengths = []
        self.log_file_path = None
        self.csv_writer = None
        self.csv_file = None

        # Ensure the metric output directory exists
        os.makedirs(metric_output_dir, exist_ok=True)

        # Create full log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_filename_prefix}_{timestamp}.csv"
        self.log_file_path = os.path.join(metric_output_dir, log_filename)
        print(f"📊 Logging metrics to: {self.log_file_path}")

        self.setup_logging(self.log_file_path)
    # --- --------------------------------------------- ---

    def setup_logging(self, log_file):
        """Initialize the CSV log file with headers."""
        try:
            self.csv_file = open(log_file, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Step',
                'Bus_Avg_Wait', 'Bus_Avg_Speed', 'Bus_Count',
                'Other_Avg_Wait', 'Other_Avg_Speed', 'Other_Count',
                'All_Avg_Wait', 'All_Avg_Speed', 'All_Count',
                'Avg_Queue_Length'
            ])
        except IOError as e:
            print(f"Error opening log file {log_file}: {e}")
            self.csv_writer = None # Ensure logging is disabled if file fails

    def collect_step_data(self, step):
        """Collect data for the current simulation step."""
        # Reset step metrics
        for vtype in self.step_metrics:
            self.step_metrics[vtype]['waiting_time'] = 0
            self.step_metrics[vtype]['speed'] = 0
            self.step_metrics[vtype]['count'] = 0

        current_queues = []
        # Get all vehicle IDs
        vehicle_ids = traci.vehicle.getIDList()

        # Collect data for each vehicle
        for veh_id in vehicle_ids:
            vtype = traci.vehicle.getTypeID(veh_id)
            waiting_time = traci.vehicle.getAccumulatedWaitingTime(veh_id)
            speed = traci.vehicle.getSpeed(veh_id)

            category = 'bus' if vtype.lower().startswith('bus') else 'other'

            self.step_metrics[category]['waiting_time'] += waiting_time
            self.step_metrics[category]['speed'] += speed
            self.step_metrics[category]['count'] += 1

            self.step_metrics['all']['waiting_time'] += waiting_time
            self.step_metrics['all']['speed'] += speed
            self.step_metrics['all']['count'] += 1

        # Collect queue lengths from all lanes
        current_queues = []
        for edge_id in traci.edge.getIDList():
            num_lanes = traci.edge.getLaneNumber(edge_id)
            for i in range(num_lanes):
                lane_id = f"{edge_id}_{i}"
                queue_length = traci.lane.getLastStepHaltingNumber(lane_id)
                if queue_length > 0:
                    current_queues.append(queue_length)

        avg_queue = np.mean(current_queues) if current_queues else 0
        self.queue_lengths.append(avg_queue)

        # Calculate averages and store in metrics
        for vtype in self.step_metrics:
            count = self.step_metrics[vtype]['count']
            if count > 0:
                avg_wait = self.step_metrics[vtype]['waiting_time'] / count
                avg_speed = self.step_metrics[vtype]['speed'] / count
                self.metrics[vtype]['waiting_times'].append(avg_wait)
                self.metrics[vtype]['speeds'].append(avg_speed)
            else:
                self.metrics[vtype]['waiting_times'].append(0)
                self.metrics[vtype]['speeds'].append(0)

        # Log to CSV if enabled and writer is valid
        if self.csv_writer:
            try:
                self.csv_writer.writerow([
                    step,
                    self.metrics['bus']['waiting_times'][-1],
                    self.metrics['bus']['speeds'][-1],
                    self.step_metrics['bus']['count'],
                    self.metrics['other']['waiting_times'][-1],
                    self.metrics['other']['speeds'][-1],
                    self.step_metrics['other']['count'],
                    self.metrics['all']['waiting_times'][-1],
                    self.metrics['all']['speeds'][-1],
                    self.step_metrics['all']['count'],
                    avg_queue
                ])
            except Exception as e:
                print(f"Error writing to CSV log at step {step}: {e}")


    def get_summary(self):
        """Return a summary of all collected metrics."""
        summary = {}
        for vtype in self.metrics:
            wt_valid = [t for t in self.metrics[vtype]['waiting_times'] if t is not None]
            s_valid = [s for s in self.metrics[vtype]['speeds'] if s is not None]
            summary[vtype] = {
                'avg_waiting_time': np.mean(wt_valid) if wt_valid else 0,
                'avg_speed': np.mean(s_valid) if s_valid else 0,
                'max_waiting_time': np.max(wt_valid) if wt_valid else 0,
                'min_speed': np.min(s_valid) if s_valid else 0 # Min speed might not be very informative
            }

        q_valid = [q for q in self.queue_lengths if q is not None]
        summary['avg_queue_length'] = np.mean(q_valid) if q_valid else 0
        summary['max_queue_length'] = np.max(q_valid) if q_valid else 0

        return summary

    def close(self):
        """Close the log file if it's open."""
        if self.csv_file:
            try:
                self.csv_file.close()
                print(f"Closed log file: {self.log_file_path}")
            except Exception as e:
                print(f"Error closing log file {self.log_file_path}: {e}")
        self.csv_file = None
        self.csv_writer = None


# Removed get_vehicle_weights as weights are now scenario-based

# --- Modified run_fixed_time ---
def run_fixed_time(config_file_path, metric_output_dir, bus_weight, max_steps=3600, seed=42):
    """
    Runs SUMO in GUI mode with fixed-time traffic lights.
    Uses scenario-specific paths and bus weight.
    """
    set_seed(seed)

    # Initialize metrics collector with the correct output directory
    metrics = MetricsCollector(metric_output_dir, log_filename_prefix="fixed_time_metrics")

    env = None # Initialize env to None
    try:
        env = SumoEnvironment(
            config_file=config_file_path,
            use_gui=True, # Fixed time usually run with GUI for visualization
            num_seconds=max_steps,
            delta_time=5 # Assuming fixed delta time
        )
        # --- Manually set bus weight ---
        env.vType_weights["bus"] = bus_weight
        # --- ----------------------- ---

        states = env.reset()
        # --- Manually set bus weight AGAIN after reset ---
        env.vType_weights["bus"] = bus_weight
        # --- --------------------------------------- ---

        done = False
        step_count = 0
        total_reward = 0.0 # Track reward even in fixed time for comparison basis

        print("\n🚦 Running fixed-time traffic control simulation...")
        print("⏱️  Press Ctrl+C in the terminal (not GUI) to stop early")

        pbar = tqdm(total=max_steps, desc="Simulating Fixed-Time", unit="step")
        while not done:
            step_count += env.delta_time # Increment by simulation step size

            # Collect metrics for this step
            metrics.collect_step_data(env.sim_step) # Use env.sim_step

            # Fixed time: Don't select actions, just step the environment
            # The environment's internal logic handles phase changes based on SUMO's fixed program
            # We still need to call env.step to advance simulation and get rewards/dones
            # Pass dummy actions (or current phases if needed by env.step implementation)
            actions = {}
            for tl_id in env.traffic_lights:
                actions[tl_id] = env.traffic_light_states[tl_id]['current_phase']

            next_states, rewards, dones, _ = env.step(actions)

            if rewards:
                step_reward = sum(rewards.values()) / len(rewards)
                total_reward += step_reward

            if any(dones.values()) or (step_count >= max_steps):
                done = True

            states = next_states
            pbar.update(env.delta_time)

    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
    except traci.exceptions.TraCIException as e:
         print(f"\n❌ TraCI Error during fixed-time run: {e}")
         # Attempt graceful shutdown
    except Exception as e:
         print(f"\n❌ Unexpected Error during fixed-time run: {e}")
    finally:
        pbar.close()
        if env:
            env.close()
        metrics.close() # Ensure metrics file is closed

    # Get and print summary statistics
    summary = metrics.get_summary()

    print("\n📊 Fixed-time Simulation Results")
    print("==============================")
    print(f"Total steps completed: {step_count}/{max_steps}")
    print(f"Total average reward (for comparison): {total_reward:.2f}")
    print("\nPerformance Metrics:")
    print("-------------------")
    print(f"  Buses: Avg Wait={summary['bus']['avg_waiting_time']:.2f}s, Avg Speed={summary['bus']['avg_speed']:.2f}m/s")
    print(f"  Other: Avg Wait={summary['other']['avg_waiting_time']:.2f}s, Avg Speed={summary['other']['avg_speed']:.2f}m/s")
    print(f"  All:   Avg Wait={summary['all']['avg_waiting_time']:.2f}s, Avg Speed={summary['all']['avg_speed']:.2f}m/s")
    print(f"  Avg Queue Length: {summary['avg_queue_length']:.2f} vehicles")
    print(f"  Max Queue Length: {summary['max_queue_length']:.2f} vehicles")

    return summary, metrics.log_file_path # Return the actual log file path
# --- --------------------------- ---


# --- Modified run_trained_model ---
def run_trained_model(config_file_path, metric_output_dir, models_folder, bus_weight, max_steps=3600, seed=42):
    """
    Load a trained RL model and run with SUMO GUI.
    Uses scenario-specific paths and bus weight.
    """
    set_seed(seed)

    # Initialize metrics collector
    metrics = MetricsCollector(metric_output_dir, log_filename_prefix="trained_model_metrics")

    env = None
    try:
        env = SumoEnvironment(
            config_file=config_file_path,
            use_gui=True, # Testing usually done with GUI
            num_seconds=max_steps,
            delta_time=5 # Assuming fixed delta time
        )
        # --- Manually set bus weight ---
        env.vType_weights["bus"] = bus_weight
        # --- ----------------------- ---

        intersection_groups = create_intersection_groups(env.intersection_types)
        state_dims, action_dims = create_state_action_dims(env)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")

        agent = SharedNoisyDoubleDQNAgent(
            state_dims=state_dims,
            action_dims=action_dims,
            intersection_groups=intersection_groups,
            # Hyperparameters below don't matter much for testing if loading a trained model,
            # but need to be provided to the constructor.
            learning_rate=1e-4,
            gamma=0.99,
            initial_sigma=0.01, # Use low sigma for testing (more greedy)
            sigma_decay=1.0,    # No decay during testing
            sigma_min=0.01,
            target_update=10000, # Effectively no target updates during test
            device=device
        )

        # Load the trained models
        try:
            agent.load_models(models_folder)
            print(f"✅ Loaded agent models from: {models_folder}")
        except FileNotFoundError:
            print(f"❌ Error: Model files not found in {models_folder}")
            print("Please ensure the model folder path is correct and contains the necessary .pt files.")
            return None, None # Return None if models can't be loaded
        except Exception as e:
            print(f"❌ Error loading models from {models_folder}: {e}")
            return None, None


        states = env.reset()
         # --- Manually set bus weight AGAIN after reset ---
        env.vType_weights["bus"] = bus_weight
        # --- --------------------------------------- ---

        done = False
        step_count = 0
        total_reward = 0.0

        print("\n🚦 Running trained model traffic control simulation...")
        print("⏱️  Press Ctrl+C in the terminal (not GUI) to stop early")

        pbar = tqdm(total=max_steps, desc="Simulating Trained Model", unit="step")
        while not done:
            step_count += env.delta_time

            # Collect metrics
            metrics.collect_step_data(env.sim_step)

            # Select actions using the loaded agent
            actions = {}
            for tl_id in env.traffic_lights:
                 if tl_id in states:
                     group_id = get_intersection_group(tl_id, intersection_groups)
                     # Use agent's greedy action for testing
                     action = agent.select_action(tl_id, states[tl_id], group_id)
                     actions[tl_id] = action
                 else:
                     current_phase = env.traffic_light_states.get(tl_id, {}).get('current_phase', 0)
                     actions[tl_id] = current_phase


            next_states, rewards, dones, _ = env.step(actions)

            if rewards:
                step_reward = sum(rewards.values()) / len(rewards)
                total_reward += step_reward

            if any(dones.values()) or (step_count >= max_steps):
                done = True

            states = next_states
            pbar.update(env.delta_time)

    except KeyboardInterrupt:
        print("\n⚠️ Simulation interrupted by user")
    except traci.exceptions.TraCIException as e:
         print(f"\n❌ TraCI Error during trained model run: {e}")
    except Exception as e:
         print(f"\n❌ Unexpected Error during trained model run: {e}")
    finally:
        pbar.close()
        if env:
            env.close()
        metrics.close()

    # Get and print summary statistics
    summary = metrics.get_summary()

    print("\n📊 Trained Model Simulation Results")
    print("=================================")
    print(f"Total steps completed: {step_count}/{max_steps}")
    print(f"Total average reward: {total_reward:.2f}")
    print("\nPerformance Metrics:")
    print("-------------------")
    print(f"  Buses: Avg Wait={summary['bus']['avg_waiting_time']:.2f}s, Avg Speed={summary['bus']['avg_speed']:.2f}m/s")
    print(f"  Other: Avg Wait={summary['other']['avg_waiting_time']:.2f}s, Avg Speed={summary['other']['avg_speed']:.2f}m/s")
    print(f"  All:   Avg Wait={summary['all']['avg_waiting_time']:.2f}s, Avg Speed={summary['all']['avg_speed']:.2f}m/s")
    print(f"  Avg Queue Length: {summary['avg_queue_length']:.2f} vehicles")
    print(f"  Max Queue Length: {summary['max_queue_length']:.2f} vehicles")

    return summary, metrics.log_file_path
# --- ---------------------------- ---


def compare_results(fixed_time_summary, model_summary, metric_output_dir):
    """Compare and display results, save comparison to the metric directory."""
    print("\n📈 Comparison: Fixed-Time vs. Trained Model")
    print("=========================================")

    # Check if summaries are valid
    if fixed_time_summary is None or model_summary is None:
        print("Comparison skipped due to missing results.")
        return

    # Calculate percentage improvements safely
    def percentage_change(old, new):
        if old is None or new is None: return "N/A"
        if old == 0: return "Inf" if new > 0 else "0.00%" # Avoid division by zero
        change = ((new - old) / abs(old)) * 100
        return f"{change:.2f}%"

    def improvement(old, new, lower_is_better=True):
         if old is None or new is None: return "N/A"
         if old == 0: return "Inf" if new != 0 else "0.00%"
         # Improvement = (Old - New) / Old * 100 if lower is better
         # Improvement = (New - Old) / Old * 100 if higher is better
         if lower_is_better:
              imp = ((old - new) / abs(old)) * 100
         else: # Higher is better (e.g., speed)
              imp = ((new - old) / abs(old)) * 100
         return f"{imp:.2f}%"


    wait_imp_bus = improvement(fixed_time_summary['bus']['avg_waiting_time'], model_summary['bus']['avg_waiting_time'])
    wait_imp_other = improvement(fixed_time_summary['other']['avg_waiting_time'], model_summary['other']['avg_waiting_time'])
    wait_imp_all = improvement(fixed_time_summary['all']['avg_waiting_time'], model_summary['all']['avg_waiting_time'])

    speed_imp_bus = improvement(fixed_time_summary['bus']['avg_speed'], model_summary['bus']['avg_speed'], lower_is_better=False)
    speed_imp_other = improvement(fixed_time_summary['other']['avg_speed'], model_summary['other']['avg_speed'], lower_is_better=False)
    speed_imp_all = improvement(fixed_time_summary['all']['avg_speed'], model_summary['all']['avg_speed'], lower_is_better=False)

    queue_imp = improvement(fixed_time_summary['avg_queue_length'], model_summary['avg_queue_length'])

    # Print comparison table
    print("\nMetric                  | Fixed-Time | Trained Model | Improvement")
    print("------------------------|------------|---------------|------------")
    print(f"Bus Waiting Time (s)    | {fixed_time_summary['bus']['avg_waiting_time']:<10.2f} | {model_summary['bus']['avg_waiting_time']:<13.2f} | {wait_imp_bus:>10}")
    print(f"Other Waiting Time (s)  | {fixed_time_summary['other']['avg_waiting_time']:<10.2f} | {model_summary['other']['avg_waiting_time']:<13.2f} | {wait_imp_other:>10}")
    print(f"All Waiting Time (s)    | {fixed_time_summary['all']['avg_waiting_time']:<10.2f} | {model_summary['all']['avg_waiting_time']:<13.2f} | {wait_imp_all:>10}")
    print(f"Bus Speed (m/s)         | {fixed_time_summary['bus']['avg_speed']:<10.2f} | {model_summary['bus']['avg_speed']:<13.2f} | {speed_imp_bus:>10}")
    print(f"Other Speed (m/s)       | {fixed_time_summary['other']['avg_speed']:<10.2f} | {model_summary['other']['avg_speed']:<13.2f} | {speed_imp_other:>10}")
    print(f"All Speed (m/s)         | {fixed_time_summary['all']['avg_speed']:<10.2f} | {model_summary['all']['avg_speed']:<13.2f} | {speed_imp_all:>10}")
    print(f"Avg Queue Length        | {fixed_time_summary['avg_queue_length']:<10.2f} | {model_summary['avg_queue_length']:<13.2f} | {queue_imp:>10}")

    # Write comparison to file in the metric directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = os.path.join(metric_output_dir, f"comparison_results_{timestamp}.txt")

    try:
        with open(comparison_file, 'w') as f:
            f.write("Comparison: Fixed-Time vs. Trained Model\n")
            f.write("=========================================\n\n")
            f.write("Metric                  | Fixed-Time | Trained Model | Improvement\n")
            f.write("------------------------|------------|---------------|------------\n")
            f.write(f"Bus Waiting Time (s)    | {fixed_time_summary['bus']['avg_waiting_time']:<10.2f} | {model_summary['bus']['avg_waiting_time']:<13.2f} | {wait_imp_bus:>10}\n")
            f.write(f"Other Waiting Time (s)  | {fixed_time_summary['other']['avg_waiting_time']:<10.2f} | {model_summary['other']['avg_waiting_time']:<13.2f} | {wait_imp_other:>10}\n")
            f.write(f"All Waiting Time (s)    | {fixed_time_summary['all']['avg_waiting_time']:<10.2f} | {model_summary['all']['avg_waiting_time']:<13.2f} | {wait_imp_all:>10}\n")
            f.write(f"Bus Speed (m/s)         | {fixed_time_summary['bus']['avg_speed']:<10.2f} | {model_summary['bus']['avg_speed']:<13.2f} | {speed_imp_bus:>10}\n")
            f.write(f"Other Speed (m/s)       | {fixed_time_summary['other']['avg_speed']:<10.2f} | {model_summary['other']['avg_speed']:<13.2f} | {speed_imp_other:>10}\n")
            f.write(f"All Speed (m/s)         | {fixed_time_summary['all']['avg_speed']:<10.2f} | {model_summary['all']['avg_speed']:<13.2f} | {speed_imp_all:>10}\n")
            f.write(f"Avg Queue Length        | {fixed_time_summary['avg_queue_length']:<10.2f} | {model_summary['avg_queue_length']:<13.2f} | {queue_imp:>10}\n")
        print(f"\n✅ Comparison saved to: {comparison_file}")
    except IOError as e:
        print(f"Error writing comparison file {comparison_file}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SUMO testing for Acosta or Pasubio with fixed-time or a trained RL model.")
    # Keep relevant args, remove paths determined by scenario
    parser.add_argument("--max_steps", type=int, default=3600, help="Max steps (seconds) per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    # Add argument for model folder, relative to scenario dir or absolute
    parser.add_argument("--models_folder", type=str, default=None, help="Path to the folder containing trained model files (.pt). If None, uses default path for scenario.")

    args = parser.parse_args()

    # Select Scenario
    selected_scenario = select_scenario()
    scenario_config = scenario_configs[selected_scenario]
    scenario_dir = scenario_config['dir']
    config_file_path = os.path.join(scenario_dir, 'run.sumocfg')
    metric_output_dir = os.path.join(scenario_dir, scenario_config['metric_subdir'])
    bus_weight = scenario_config['bus_weight']

    # Determine model folder path
    models_folder_path = args.models_folder
    if models_folder_path is None:
        # Use default path if not provided
        models_folder_path = os.path.join(scenario_dir, scenario_config['model_subdir'])
        print(f"Using default model folder: {models_folder_path}")
    elif not os.path.isabs(models_folder_path):
        # Make relative path absolute based on scenario directory
        models_folder_path = os.path.abspath(os.path.join(scenario_dir, models_folder_path))
        print(f"Using relative model folder, resolved to: {models_folder_path}")
    else:
        print(f"Using absolute model folder: {models_folder_path}")

    # Check if the resolved model folder exists before proceeding
    if not os.path.isdir(models_folder_path):
         print(f"❌ Error: Specified model folder does not exist: {models_folder_path}")
         sys.exit(1)


    print(f"\n--- Testing Scenario: {selected_scenario} ---")
    print(f"Config File: {config_file_path}")
    print(f"Metrics Dir: {metric_output_dir}")
    print(f"Models Folder: {models_folder_path}")
    print(f"Bus Weight: {bus_weight}")


    print("\nSelect Mode:")
    print(" 1) Fixed-time run (no RL)")
    print(" 2) Load a trained RL model")
    print(" 3) Compare both (run fixed-time then trained model)")

    choice = input("Enter '1', '2', or '3': ").strip()

    fixed_time_summary, fixed_time_log = None, None
    model_summary, model_log = None, None

    if choice == "1":
        print("\n✅ Running FIXED-TIME mode.")
        fixed_time_summary, fixed_time_log = run_fixed_time(
            config_file_path=config_file_path,
            metric_output_dir=metric_output_dir,
            bus_weight=bus_weight,
            max_steps=args.max_steps,
            seed=args.seed
        )

    elif choice == "2":
        print("\n✅ Running TRAINED MODEL mode.")
        model_summary, model_log = run_trained_model(
            config_file_path=config_file_path,
            metric_output_dir=metric_output_dir,
            models_folder=models_folder_path,
            bus_weight=bus_weight,
            max_steps=args.max_steps,
            seed=args.seed
        )

    elif choice == "3":
        print("\n✅ Running COMPARISON mode.")

        # Run fixed-time first
        print("\n--- Running fixed-time simulation first ---")
        fixed_time_summary, fixed_time_log = run_fixed_time(
            config_file_path=config_file_path,
            metric_output_dir=metric_output_dir,
            bus_weight=bus_weight,
            max_steps=args.max_steps,
            seed=args.seed
        )

        # Run trained model
        print("\n--- Now running trained model simulation ---")
        model_summary, model_log = run_trained_model(
            config_file_path=config_file_path,
            metric_output_dir=metric_output_dir,
            models_folder=models_folder_path,
            bus_weight=bus_weight,
            max_steps=args.max_steps,
            seed=args.seed
        )

        # Compare results
        compare_results(fixed_time_summary, model_summary, metric_output_dir)

        print(f"\n📊 Detailed logs saved to:")
        if fixed_time_log: print(f"   - Fixed-time: {fixed_time_log}")
        if model_log: print(f"   - Trained model: {model_log}")

    else:
        print("❌ Invalid choice. Please run again and pick '1', '2', or '3'.")

    print(f"\n--- Testing finished for {selected_scenario} ---")
