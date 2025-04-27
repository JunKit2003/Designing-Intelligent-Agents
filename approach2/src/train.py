import os
import sys # Import sys
import time
import numpy as np
import torch
import traci
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
import subprocess
from sumo_env import SumoEnvironment
from replay_buffer import ReplayBuffer
from dqn_agent import SharedNoisyDoubleDQNAgent
from utils import (
    set_seed, create_intersection_groups, get_intersection_group,
    create_state_action_dims, save_metrics
)

# --- Scenario Configuration (Used for paths and TEST weights) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate up two levels to get the base directory containing both scenario folders
base_dir = os.path.abspath(os.path.join(script_dir, '..'))

scenario_configs = {
    'acosta': {
        'dir': os.path.join(base_dir, 'acosta'),
        'bus_weight': 1.0,
        'output_subdir': 'output_acosta'
    },
    'pasubio': {
        'dir': os.path.join(base_dir, 'pasubio'),
        'bus_weight': 1.0,
        'output_subdir': 'output_pasubio'
    }
}

DEFAULT_BUS_WEIGHT_TRAINING = 1.0

# Traffic level configuration (remains the same)
traffic_levels = {
    "light": { "period": 3.0, "fringe_factor": 5.0, "depart_speed": "random", "min_distance": 300, "max_distance": 2000, },
    "medium": { "period": 1.5, "fringe_factor": 10.0, "depart_speed": "random", "min_distance": 300, "max_distance": 3000, },
    "heavy": { "period": 0.8, "fringe_factor": 15.0, "depart_speed": "random", "min_distance": 200, "max_distance": 4000, }
}

def select_scenario():
    """Prompts the user to select a scenario."""
    while True:
        print("\nSelect Scenario (for file paths and output location):")
        print("  1) Acosta")
        print("  2) Pasubio")
        choice = input("Enter '1' or '2': ").strip()
        if choice == '1':
            return 'acosta'
        elif choice == '2':
            return 'pasubio'
        else:
            print("Invalid choice. Please enter '1' or '2'.")

def generate_mixed_traffic_route_file(scenario_dir, output_file, max_steps):
    """
    Generates a route file with mixed traffic levels for the specified scenario.
    Paths are now relative to the scenario directory.

    Args:
        scenario_dir (str): Path to the root directory of the selected scenario.
        output_file (str): Path to save the generated route file (within scenario_dir).
        max_steps (int): Maximum simulation steps (in seconds).
    """
    original_config_file = os.path.join(scenario_dir, 'run.sumocfg') # Path within scenario
    # Parse the original config file to get necessary info (net-file)
    try:
        tree = ET.parse(original_config_file)
        root = tree.getroot()
        # Make net_file path absolute based on scenario_dir
        net_file_rel = root.find(".//net-file").get("value")
        net_file_abs = os.path.abspath(os.path.join(scenario_dir, net_file_rel))

        # Extract additional parameters (make path absolute)
        add_file_elem = root.find(".//additional-file")
        add_file_abs = ""
        if add_file_elem is not None:
            add_file_rel = add_file_elem.get("value")
            # Handle multiple additional files if needed
            add_files_rel = add_file_rel.split(' ') # Split by space if multiple files
            add_file_abs_list = [os.path.abspath(os.path.join(scenario_dir, f)) for f in add_files_rel]
            add_file_abs = " ".join(add_file_abs_list) # Rejoin for command line arg

    except FileNotFoundError:
        print(f"Error: Config file not found at {original_config_file}")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing config file {original_config_file}: {e}")
        sys.exit(1)
    except AttributeError:
         print(f"Error: Could not find 'net-file' element in {original_config_file}")
         sys.exit(1)


    # Create temporary route files (use scenario dir for temp files)
    temp_route_files = {}
    for level, params in traffic_levels.items():
        temp_file = os.path.join(scenario_dir, f"temp_route_{level}.rou.xml") # Temp file in scenario dir
        temp_route_files[level] = temp_file

        # Generate route file for this traffic level
        command = [
            sys.executable, # Use the current Python interpreter
            os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
            "-n", net_file_abs, # Use absolute path
            "-r", temp_file,
            "--begin", "0",
            "--end", str(max_steps),
            "--period", str(params["period"]),
            "--fringe-factor", str(params["fringe_factor"]),
            "--min-distance", str(params["min_distance"]),
            "--max-distance", str(params["max_distance"]),
            "--validate",
            "--random-depart"
        ]

        if add_file_abs:
             # randomTrips.py might expect multiple -a flags or comma-separated
             # Assuming comma-separated works, adjust if needed based on tool's usage
             command.extend(["-a", add_file_abs.replace(" ", ",")])

        try:
            # Run from the scenario directory to resolve relative paths within config if any
            # print(f"Generating temporary route file for level '{level}'...") # Less verbose
            subprocess.run(command, check=True, cwd=scenario_dir, capture_output=True, text=True) # Capture output
            # print(f"Successfully generated {temp_file}") # Less verbose
        except subprocess.CalledProcessError as e:
            print(f"Error running randomTrips.py for level '{level}':")
            print(f"Command: {' '.join(e.cmd)}")
            print(f"Return Code: {e.returncode}")
            print(f"Output:\n{e.stdout}")
            print(f"Error Output:\n{e.stderr}")
            # Decide whether to exit or continue
            # sys.exit(1) # Exit if route generation fails
            print("Continuing without this traffic level...") # Or continue
        except FileNotFoundError:
            print(f"Error: 'python' or 'randomTrips.py' not found. Ensure SUMO_HOME is set correctly and Python is in PATH.")
            sys.exit(1)


    # Now merge the route files
    all_vehicles = []
    vehicle_id_counter = 0

    for level, temp_file in temp_route_files.items():
        if os.path.exists(temp_file):
            try:
                tree = ET.parse(temp_file)
                root = tree.getroot()
                for vehicle in root.findall(".//vehicle"):
                    depart_time = float(vehicle.get("depart"))
                    vehicle.set("id", f"{level}_{vehicle_id_counter}")
                    vehicle_id_counter += 1
                    all_vehicles.append((depart_time, vehicle))
            except ET.ParseError as e:
                print(f"Warning: Could not parse {temp_file}: {e}")
            finally:
                 # Clean up temporary file immediately after parsing
                 try:
                      os.remove(temp_file)
                 except OSError as e:
                      print(f"Warning: Could not remove temp file {temp_file}: {e}")


    all_vehicles.sort(key=lambda x: x[0])

    # Create a new route file with all vehicles
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        
        # Add vehicle types if needed
        f.write('    <vType id="passenger" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70" speedDev="0.1" />\n')
        
        # Add all vehicles
        for _, vehicle in all_vehicles:
            vehicle_xml = ET.tostring(vehicle, encoding='unicode')
            f.write(f'    {vehicle_xml}\n')
        
        f.write('</routes>\n')
    
    # Clean up temporary files
    for temp_file in temp_route_files.values():
        if os.path.exists(temp_file):
            os.remove(temp_file)


def update_config_with_route(scenario_dir, new_route_file_rel, output_config_file):
    """
    Updates the SUMO config file to use the newly generated route file.
    Uses relative path for the new route file within the scenario config.

    Args:
        scenario_dir (str): Path to the root directory of the selected scenario.
        new_route_file_rel (str): Relative path to the new route file from scenario_dir.
        output_config_file (str): Path to save the modified SUMO config file (within scenario_dir).
    """
    original_config_file = os.path.join(scenario_dir, 'run.sumocfg')
    try:
        tree = ET.parse(original_config_file)
        root = tree.getroot()

        input_elem = root.find(".//input")
        if input_elem is None:
             raise ValueError("Could not find <input> element in config file.")

        # Remove existing route-files elements
        for route_files_elem in input_elem.findall("route-files"):
            input_elem.remove(route_files_elem)

        # Create a new route-files element with the relative path
        new_route_files_elem = ET.SubElement(input_elem, "route-files")
        new_route_files_elem.set("value", new_route_file_rel) # Use relative path

        # Ensure output path is correct
        if not os.path.isabs(output_config_file):
             output_config_file = os.path.join(scenario_dir, output_config_file)

        tree.write(output_config_file)
        # print(f"Updated config file: {output_config_file}")

    except FileNotFoundError:
        print(f"Error: Original config file not found at {original_config_file}")
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing config file {original_config_file}: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error processing config file {original_config_file}: {e}")
        sys.exit(1)
    except IOError as e:
        print(f"Error writing updated config file {output_config_file}: {e}")
        sys.exit(1)


def train(
    scenario_name, # Added scenario name
    num_episodes,
    max_steps,
    batch_size,
    buffer_size,
    learning_rate,
    gamma,
    target_update,
    eval_interval,
    save_interval,
    seed,
    use_gui=False,
    mixed_route_interval_length=300 # Keep this parameter, even if unused in route gen
):
    """
    Train a Shared Noisy Double DQN agent. Uses standard weights (bus=1.0)
    for training and evaluation within this script.
    """
    # --- Get Scenario Specific Config ---
    if scenario_name not in scenario_configs:
        print(f"Error: Unknown scenario '{scenario_name}'")
        sys.exit(1)
    config = scenario_configs[scenario_name]
    scenario_dir = config['dir']
    # Construct output directory path within the scenario folder
    output_dir = os.path.join(scenario_dir, config['output_subdir'])
    original_config_file = os.path.join(scenario_dir, 'run.sumocfg')
    # --- ---------------------------- ---

    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    print(f"--- Training Scenario Files: {scenario_name} ---")
    print(f"Using Config Base: {original_config_file}")
    print(f"Outputting to: {output_dir}")
    print(f"!!! Using Standard Bus Weight ({DEFAULT_BUS_WEIGHT_TRAINING}) for Training !!!") # Highlight this

    # Generate the initial mixed traffic route file (relative path for config)
    initial_route_file_rel = "route_mixed_initial.rou.xml"
    initial_route_file_abs = os.path.join(scenario_dir, initial_route_file_rel)
    mixed_config_file = os.path.join(scenario_dir, "mixed_traffic.sumocfg") # Place temp config in scenario dir

    # print("Generating initial route file...") # Less verbose
    generate_mixed_traffic_route_file(scenario_dir, initial_route_file_abs, max_steps)
    update_config_with_route(scenario_dir, initial_route_file_rel, mixed_config_file)
    # print("Initial route file generated and config updated.") # Less verbose

    # Initialize the environment with the mixed traffic route file
    env = None
    try:
        env = SumoEnvironment(
            config_file=mixed_config_file, # Use the temp config
            use_gui=use_gui,
            num_seconds=max_steps,
            delta_time=5 # Assuming delta_time is fixed, make it an arg if needed
        )
        print("SUMO Environment initialized.")
    except Exception as e:
        print(f"Error initializing SumoEnvironment: {e}")
        # Clean up generated files before exiting
        if os.path.exists(initial_route_file_abs): os.remove(initial_route_file_abs)
        if os.path.exists(mixed_config_file): os.remove(mixed_config_file)
        sys.exit(1)


    # --- Force Standard Bus Weight for Training ---
    print(f"Setting training bus weight in environment to: {DEFAULT_BUS_WEIGHT_TRAINING}")
    env.vType_weights["bus"] = DEFAULT_BUS_WEIGHT_TRAINING
    # --- -------------------------------------- ---

    intersection_groups = create_intersection_groups(env.intersection_types)
    state_dims, action_dims = create_state_action_dims(env)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    agent = SharedNoisyDoubleDQNAgent(
        state_dims=state_dims,
        action_dims=action_dims,
        intersection_groups=intersection_groups,
        learning_rate=learning_rate,
        gamma=gamma,
        initial_sigma=1.0, # Consider making this an arg
        sigma_decay=0.995, # Consider making this an arg
        sigma_min=0.01,    # Consider making this an arg
        target_update=target_update,
        device=device
    )

    replay_buffers = {
        group_id: ReplayBuffer(buffer_size) for group_id in intersection_groups
    }

    metrics = {
        "episode_rewards": [], "episode_lengths": [],
        "average_queue_lengths": [], "average_waiting_times": [], "average_speeds": [],
        "training_times": [], "eval_rewards": [], "eval_queue_lengths": [],
        "eval_waiting_times": [], "eval_speeds": []
    }

    WARMUP_STEPS = 5000 # Consider making this an arg

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ({scenario_name}) ---")

        # Generate a new mixed traffic route file for this episode
        route_file_rel = f"route_mixed_episode_{episode+1}.rou.xml"
        route_file_abs = os.path.join(scenario_dir, route_file_rel)
        generate_mixed_traffic_route_file(scenario_dir, route_file_abs, max_steps)
        update_config_with_route(scenario_dir, route_file_rel, mixed_config_file)

        # Reset environment (will use the updated mixed_config_file)
        try:
            states = env.reset()
            # --- Force Standard Bus Weight AGAIN after Reset ---
            env.vType_weights["bus"] = DEFAULT_BUS_WEIGHT_TRAINING
            # --- ------------------------------------------- ---
        except Exception as e:
            print(f"Error resetting environment for episode {episode + 1}: {e}")
            # Attempt cleanup before potentially stopping
            if os.path.exists(route_file_abs): os.remove(route_file_abs)
            if os.path.exists(mixed_config_file): os.remove(mixed_config_file)
            # Decide whether to break or continue
            break # Stop training if reset fails


        episode_reward = 0.0
        episode_length = 0
        episode_queue_lengths = []
        episode_waiting_times = []
        episode_speeds = []

        start_time = time.time()
        done = False

        pbar = tqdm(total=max_steps, desc=f"Ep {episode+1} Sim", unit="step")
        try:
            while not done:
                actions = {}
                for tl_id in env.traffic_lights:
                    group_id = get_intersection_group(tl_id, intersection_groups)
                    # Assume state exists - reverting the check
                    action = agent.select_action(tl_id, states[tl_id], group_id)
                    actions[tl_id] = action

                next_states, rewards, dones, info = env.step(actions)
                done = any(dones.values()) # Episode ends if any intersection is 'done' (usually time limit)

                # Store experience in buffer - reverting the check
                for tl_id in env.traffic_lights:
                    group_id = get_intersection_group(tl_id, intersection_groups)
                    replay_buffers[group_id].add(
                        states[tl_id],
                        actions[tl_id],
                        rewards[tl_id],
                        next_states[tl_id],
                        dones[tl_id]
                    )

                states = next_states
                # Calculate average step reward only if rewards dict is not empty
                if rewards:
                     step_reward = sum(rewards.values()) / len(rewards)
                     episode_reward += step_reward
                # Use env.sim_step for accurate tracking if delta_time varies
                # episode_length = env.sim_step # Use this if delta_time might change
                episode_length += env.delta_time # Increment by simulation step size

                # --- Metrics Calculation (per step) - Reverted Checks ---
                queue_length = 0
                waiting_time = 0
                speed_sum = 0
                lane_count = 0

                for tl_id in env.traffic_lights:
                    for lane in env.incoming_lanes.get(tl_id, []): # Use .get for safety
                        # Reverted checks for lane/tl existence in traci
                        q = traci.lane.getLastStepHaltingNumber(lane)
                        queue_length += q
                        n_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                        if n_vehicles > 0:
                            waiting_time += traci.lane.getWaitingTime(lane)
                        speed_sum += traci.lane.getLastStepMeanSpeed(lane)
                        lane_count += 1

                # Calculate averages safely
                # Use len(env.traffic_lights) as the denominator base assuming all are active
                # Or use lane_count if you want average per lane processed
                avg_queue_length = queue_length / len(env.traffic_lights) if env.traffic_lights else 0
                avg_waiting_time = waiting_time / lane_count if lane_count > 0 else 0
                avg_speed = speed_sum / lane_count if lane_count > 0 else 0

                episode_queue_lengths.append(avg_queue_length)
                episode_waiting_times.append(avg_waiting_time)
                episode_speeds.append(avg_speed)
                # --- End Metrics Calculation ---

                # Update agent
                for group_id in intersection_groups:
                    # Start training after warmup period
                    if len(replay_buffers[group_id]) > WARMUP_STEPS:
                        agent.update(group_id, replay_buffers[group_id], batch_size, episode_reward=None, episode_num=episode + 1) # Pass episode num for adaptive sigma

                pbar.update(env.delta_time)
                if episode_length >= max_steps:
                    done = True

        except traci.exceptions.TraCIException as e:
            print(f"\nTraCI Error during episode {episode + 1} at step {env.sim_step}: {e}")
            print("Attempting to continue to next episode...")
            # Mark episode as potentially incomplete or handle differently if needed
            done = True # End the current episode loop
        except KeyError as e:
             print(f"\nKeyError during episode {episode + 1} (likely missing state/action for TL {e}): ")
             print("Attempting to continue to next episode...")
             done = True
        except Exception as e:
            print(f"\nUnexpected Error during episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue to next episode...")
            done = True # End the current episode loop
        finally:
            pbar.close()


        end_time = time.time()
        training_time = end_time - start_time

        # Store episode metrics
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["average_queue_lengths"].append(np.mean(episode_queue_lengths) if episode_queue_lengths else 0)
        metrics["average_waiting_times"].append(np.mean(episode_waiting_times) if episode_waiting_times else 0)
        metrics["average_speeds"].append(np.mean(episode_speeds) if episode_speeds else 0)
        metrics["training_times"].append(training_time)

        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Length: {episode_length} steps ({env.sim_step} sim seconds)")
        print(f"  Avg Queue: {metrics['average_queue_lengths'][-1]:.2f}")
        print(f"  Avg Wait : {metrics['average_waiting_times'][-1]:.2f}")
        print(f"  Avg Speed: {metrics['average_speeds'][-1]:.2f}")
        print(f"  Training Time: {training_time:.2f}s")

        for group_id in intersection_groups:
            agent.update(group_id, replay_buffers[group_id], batch_size,
                        episode_reward=episode_reward, episode_num=episode + 1)
        
        any_gid = list(intersection_groups.keys())[0]
        print(f" [Episode {episode + 1}] Total EpReward: {episode_reward:.2f} | Sigma: {agent.agents[any_gid].sigma:.4f}")
        
        # Evaluation - Use standard weight for consistency during training eval
        if (episode + 1) % eval_interval == 0:
            print("  Running evaluation (using standard training weights)...")
            try:
                # Pass DEFAULT_BUS_WEIGHT_TRAINING to evaluate
                eval_metrics = evaluate(env, agent, intersection_groups, DEFAULT_BUS_WEIGHT_TRAINING)
                metrics["eval_rewards"].append(eval_metrics["reward"])
                metrics["eval_queue_lengths"].append(eval_metrics["queue_length"])
                metrics["eval_waiting_times"].append(eval_metrics["waiting_time"])
                metrics["eval_speeds"].append(eval_metrics["speed"])

                print("  Evaluation Results:")
                print(f"    Reward: {eval_metrics['reward']:.2f}")
                print(f"    Avg Queue: {eval_metrics['queue_length']:.2f}")
                print(f"    Avg Wait : {eval_metrics['waiting_time']:.2f}")
                print(f"    Avg Speed: {eval_metrics['speed']:.2f}")
            except Exception as e:
                print(f"Error during evaluation for episode {episode + 1}: {e}")
                # Append placeholder values or handle as needed
                metrics["eval_rewards"].append(None)
                metrics["eval_queue_lengths"].append(None)
                metrics["eval_waiting_times"].append(None)
                metrics["eval_speeds"].append(None)


        # Save models and metrics
        if (episode + 1) % save_interval == 0:
            print(f"  Saving models and metrics to {output_dir}...")
            try:
                agent.save_models(os.path.join(output_dir, "model", f"models_episode_{episode + 1}"))
                save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
                plot_metrics(metrics, output_dir)
                print("  Save complete.")
            except Exception as e:
                 print(f"Error during saving for episode {episode + 1}: {e}")


        # Clean up old route files (relative to scenario_dir)
        old_route_file = os.path.join(scenario_dir, f"route_mixed_episode_{episode}.rou.xml")
        if episode > 0 and os.path.exists(old_route_file):
            try:
                os.remove(old_route_file)
            except OSError as e:
                print(f"Warning: Could not remove old route file {old_route_file}: {e}")

        # Clean up the initial route file after the first episode
        if episode == 0 and os.path.exists(initial_route_file_abs):
             try:
                os.remove(initial_route_file_abs)
             except OSError as e:
                print(f"Warning: Could not remove initial route file {initial_route_file_abs}: {e}")


    print(f"\n--- Training finished for {scenario_name} ---")
    if env:
        env.close()

    # Save final models and metrics
    print(f"Saving final models and metrics to {output_dir}...")
    try:
        agent.save_models(os.path.join(output_dir, "models_final"))
        save_metrics(metrics, os.path.join(output_dir, "metrics.json"))
        plot_metrics(metrics, output_dir)
        print("Final save complete.")
    except Exception as e:
         print(f"Error during final saving: {e}")


    # Clean up the last generated route file and the temp config
    last_route_file = os.path.join(scenario_dir, f"route_mixed_episode_{num_episodes}.rou.xml")
    if os.path.exists(last_route_file):
        try:
            os.remove(last_route_file)
        except OSError as e:
            print(f"Warning: Could not remove last route file {last_route_file}: {e}")
    if os.path.exists(mixed_config_file):
        try:
            os.remove(mixed_config_file)
        except OSError as e:
            print(f"Warning: Could not remove temp config file {mixed_config_file}: {e}")


def evaluate(env, agent, intersection_groups, bus_weight, num_episodes=3):
    """Evaluation function. Resets env and sets the specified bus weight."""
    total_reward = 0.0
    total_queue_length = 0.0
    total_waiting_time = 0.0
    total_speed = 0.0
    max_steps = env.num_seconds # Get max steps from env

    for i in range(num_episodes):
        try:
            states = env.reset()
            # --- Set the specified bus weight for this evaluation episode ---
            env.vType_weights["bus"] = bus_weight
            # --- ------------------------------------------------------- ---
        except Exception as e:
            print(f"Error resetting environment for evaluation episode {i+1}: {e}")
            continue # Skip this evaluation episode


        ep_reward = 0.0
        ep_queue_lengths = []
        ep_waiting_times = []
        ep_speeds = []
        done = False
        step_count = 0

        try:
            while not done:
                actions = {}
                for tl_id in env.traffic_lights:
                     # Assume state exists for evaluation
                     group_id = get_intersection_group(tl_id, intersection_groups)
                     action = agent.select_action(tl_id, states[tl_id], group_id) # Use agent's greedy action
                     actions[tl_id] = action

                next_states, rewards, dones, _ = env.step(actions)
                done = any(dones.values())
                states = next_states
                if rewards:
                     ep_reward += sum(rewards.values()) / len(rewards) # Average reward per step
                step_count += env.delta_time

                # --- Metrics Calculation (Reverted Checks) ---
                queue_length = 0
                waiting_time = 0
                speed_sum = 0
                lane_count = 0
                for tl_id in env.traffic_lights:
                    for lane in env.incoming_lanes.get(tl_id, []):
                        q = traci.lane.getLastStepHaltingNumber(lane)
                        queue_length += q
                        waiting_time += traci.lane.getWaitingTime(lane)
                        speed_sum += traci.lane.getLastStepMeanSpeed(lane)
                        lane_count += 1

                avg_queue_length = queue_length / len(env.traffic_lights) if env.traffic_lights else 0
                avg_waiting_time = waiting_time / lane_count if lane_count > 0 else 0
                avg_speed = speed_sum / lane_count if lane_count > 0 else 0
                ep_queue_lengths.append(avg_queue_length)
                ep_waiting_times.append(avg_waiting_time)
                ep_speeds.append(avg_speed)
                # --- End Metrics ---

                if step_count >= max_steps:
                    done = True

        except traci.exceptions.TraCIException as e:
            print(f"\nTraCI Error during evaluation episode {i+1}: {e}")
            # Mark evaluation as potentially failed for this episode
            ep_reward, ep_queue_lengths, ep_waiting_times, ep_speeds = 0, [], [], [] # Reset metrics
        except KeyError as e:
             print(f"\nKeyError during evaluation episode {i+1} (likely missing state/action for TL {e})")
             ep_reward, ep_queue_lengths, ep_waiting_times, ep_speeds = 0, [], [], []
        except Exception as e:
            print(f"\nUnexpected Error during evaluation episode {i+1}: {e}")
            import traceback
            traceback.print_exc()
            ep_reward, ep_queue_lengths, ep_waiting_times, ep_speeds = 0, [], [], []


        total_reward += ep_reward
        total_queue_length += np.mean(ep_queue_lengths) if ep_queue_lengths else 0
        total_waiting_time += np.mean(ep_waiting_times) if ep_waiting_times else 0
        total_speed += np.mean(ep_speeds) if ep_speeds else 0

    avg_reward = total_reward / num_episodes
    avg_queue_length = total_queue_length / num_episodes
    avg_waiting_time = total_waiting_time / num_episodes
    avg_speed = total_speed / num_episodes

    return {
        "reward": avg_reward,
        "queue_length": avg_queue_length,
        "waiting_time": avg_waiting_time,
        "speed": avg_speed
    }

def plot_metrics(metrics, output_dir):
    """Plotting function. Saves plots to the specified output directory."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Filter out None values before plotting
    episodes = list(range(1, len(metrics["episode_rewards"]) + 1))
    valid_rewards = [(e, r) for e, r in zip(episodes, metrics["episode_rewards"]) if r is not None]
    valid_queues = [(e, q) for e, q in zip(episodes, metrics["average_queue_lengths"]) if q is not None]
    valid_waits = [(e, w) for e, w in zip(episodes, metrics["average_waiting_times"]) if w is not None]
    valid_speeds = [(e, s) for e, s in zip(episodes, metrics["average_speeds"]) if s is not None]
    valid_times = [(e, t) for e, t in zip(episodes, metrics["training_times"]) if t is not None]

    if valid_rewards:
        eps, vals = zip(*valid_rewards)
        plt.figure(figsize=(10, 6))
        plt.plot(eps, vals)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward per Episode")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "episode_rewards.png"))
        plt.close()

    if valid_queues:
        eps, vals = zip(*valid_queues)
        plt.figure(figsize=(10, 6))
        plt.plot(eps, vals)
        plt.title("Average Queue Lengths per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length (vehicles)")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "average_queue_lengths.png"))
        plt.close()

    if valid_waits:
        eps, vals = zip(*valid_waits)
        plt.figure(figsize=(10, 6))
        plt.plot(eps, vals)
        plt.title("Average Waiting Times per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Waiting Time (s)")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "average_waiting_times.png"))
        plt.close()

    if valid_speeds:
        eps, vals = zip(*valid_speeds)
        plt.figure(figsize=(10, 6))
        plt.plot(eps, vals)
        plt.title("Average Speeds per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Speed (m/s)")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "average_speeds.png"))
        plt.close()

    if valid_times:
        eps, vals = zip(*valid_times)
        plt.figure(figsize=(10, 6))
        plt.plot(eps, vals)
        plt.title("Training Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Time (s)")
        plt.grid(True)
        plt.savefig(os.path.join(fig_dir, "training_times.png"))
        plt.close()

    # Plot evaluation metrics if available and valid
    if metrics.get("eval_rewards"):
        eval_episodes = list(range(1, len(metrics["eval_rewards"]) + 1))
        eval_interval = len(episodes) // len(eval_episodes) if len(eval_episodes) > 0 else 1
        real_eval_episodes = [eval_interval * i for i in eval_episodes]

        valid_eval_rewards = [(e, r) for e, r in zip(real_eval_episodes, metrics["eval_rewards"]) if r is not None]
        valid_eval_queues = [(e, q) for e, q in zip(real_eval_episodes, metrics["eval_queue_lengths"]) if q is not None]
        valid_eval_waits = [(e, w) for e, w in zip(real_eval_episodes, metrics["eval_waiting_times"]) if w is not None]
        valid_eval_speeds = [(e, s) for e, s in zip(real_eval_episodes, metrics["eval_speeds"]) if s is not None]


        if valid_eval_rewards:
            eps, vals = zip(*valid_eval_rewards)
            plt.figure(figsize=(10, 6))
            plt.plot(eps, vals, marker='o')
            plt.title("Evaluation Rewards (Using Standard Training Weights)") # Clarify title
            plt.xlabel("Episode")
            plt.ylabel("Average Evaluation Reward")
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, "eval_rewards.png"))
            plt.close()

        if valid_eval_queues:
            eps, vals = zip(*valid_eval_queues)
            plt.figure(figsize=(10, 6))
            plt.plot(eps, vals, marker='o')
            plt.title("Evaluation Average Queue Lengths (Using Standard Training Weights)") # Clarify title
            plt.xlabel("Episode")
            plt.ylabel("Average Queue Length (vehicles)")
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, "eval_queue_lengths.png"))
            plt.close()

        if valid_eval_waits:
            eps, vals = zip(*valid_eval_waits)
            plt.figure(figsize=(10, 6))
            plt.plot(eps, vals, marker='o')
            plt.title("Evaluation Average Waiting Times (Using Standard Training Weights)") # Clarify title
            plt.xlabel("Episode")
            plt.ylabel("Average Waiting Time (s)")
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, "eval_waiting_times.png"))
            plt.close()

        if valid_eval_speeds:
            eps, vals = zip(*valid_eval_speeds)
            plt.figure(figsize=(10, 6))
            plt.plot(eps, vals, marker='o')
            plt.title("Evaluation Average Speeds (Using Standard Training Weights)") # Clarify title
            plt.xlabel("Episode")
            plt.ylabel("Average Speed (m/s)")
            plt.grid(True)
            plt.savefig(os.path.join(fig_dir, "eval_speeds.png"))
            plt.close()


if __name__ == "__main__":
    # Check if SUMO_HOME is set
    if 'SUMO_HOME' not in os.environ:
        print("Error: SUMO_HOME environment variable is not set.")
        print("Please set SUMO_HOME to the root directory of your SUMO installation.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Train a Noisy Double DQN agent with mixed traffic levels for Acosta or Pasubio.")
    # Keep hyperparameters as arguments
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to train")
    parser.add_argument("--max_steps", type=int, default=3600, help="Max steps (seconds) per episode")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=200000, help="Replay buffer capacity")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--target_update", type=int, default=10, help="Steps between target net updates")
    parser.add_argument("--eval_interval", type=int, default=10, help="Episodes between evaluations")
    parser.add_argument("--save_interval", type=int, default=20, help="Episodes between saves")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_gui", action="store_true", help="Use SUMO GUI")
    parser.add_argument("--interval_length", type=int, default=300, help="Interval length in seconds for varying traffic (e.g., 300 for 5 mins)")
    # Removed config_file and output_dir as they are determined by scenario choice

    args = parser.parse_args()

    # Select Scenario
    selected_scenario = select_scenario()

    # Start Training
    train(
        scenario_name=selected_scenario,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        target_update=args.target_update,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        seed=args.seed,
        use_gui=args.use_gui,
        mixed_route_interval_length=args.interval_length # Pass this arg
    )
