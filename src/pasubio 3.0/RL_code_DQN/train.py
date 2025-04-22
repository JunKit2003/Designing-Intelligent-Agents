import os
import time
import numpy as np
import torch
import traci
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import xml.etree.ElementTree as ET
import subprocess
import random
from sumo_env import SumoEnvironment
from replay_buffer import ReplayBuffer
from dqn_agent import SharedNoisyDoubleDQNAgent
from utils import (
    set_seed, create_intersection_groups, get_intersection_group,
    create_state_action_dims, save_metrics, moving_average
)

# Traffic level configuration
traffic_levels = {
    "light": {
        "period": 3.0,
        "fringe_factor": 5.0,
        "depart_speed": "random",
        "min_distance": 300,
        "max_distance": 2000,
    },
    "medium": {
        "period": 1.5,
        "fringe_factor": 10.0,
        "depart_speed": "random",
        "min_distance": 300,
        "max_distance": 3000,
    },
    "heavy": {
        "period": 0.8,
        "fringe_factor": 15.0,
        "depart_speed": "random",
        "min_distance": 200,
        "max_distance": 4000,
    }
}

def generate_mixed_traffic_route_file(config_file, output_file, max_steps):
    """
    Generates a route file with mixed traffic levels (light, medium, heavy)
    randomly distributed throughout the simulation period.
    
    Args:
        config_file (str): Path to the original SUMO config file.
        output_file (str): Path to save the generated route file.
        max_steps (int): Maximum simulation steps (in seconds).
    """
    # Parse the original config file to get necessary info (net-file)
    tree = ET.parse(config_file)
    root = tree.getroot()
    net_file = root.find(".//net-file").get("value")
    
    # Extract additional parameters
    add_file = root.find(".//additional-file")
    if add_file is not None:
        add_file = add_file.get("value")
    else:
        add_file = ""
    
    # Create temporary route files for each traffic level
    temp_route_files = {}
    for level, params in traffic_levels.items():
        temp_file = f"temp_route_{level}.rou.xml"
        temp_route_files[level] = temp_file
        
        # Generate route file for this traffic level
        command = [
            "python",
            os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py"),
            "-n", net_file,
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
        
        if add_file:
            command.extend(["-a", add_file])
        
        subprocess.run(command, check=True)
    
    # Now merge the route files with time-based segmentation
    # Parse each temp route file and collect all vehicle elements
    all_vehicles = []
    vehicle_id_counter = 0  # To ensure unique IDs
    
    for level, temp_file in temp_route_files.items():
        if os.path.exists(temp_file):
            try:
                tree = ET.parse(temp_file)
                root = tree.getroot()
                for vehicle in root.findall(".//vehicle"):
                    # Assign a new unique ID to each vehicle
                    depart_time = float(vehicle.get("depart"))
                    vehicle.set("id", f"{level}_{vehicle_id_counter}")
                    vehicle_id_counter += 1
                    all_vehicles.append((depart_time, vehicle))
            except ET.ParseError as e:
                print(f"Warning: Could not parse {temp_file}: {e}")
    
    # Sort vehicles by departure time
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


def update_config_with_route(original_config_file, new_route_file, output_config_file):
    """
    Updates the SUMO config file to use the newly generated route file.
    
    Args:
        original_config_file (str): Path to the original SUMO config file.
        new_route_file (str): Path to the newly generated route file.
        output_config_file (str): Path to save the modified SUMO config file.
    """
    tree = ET.parse(original_config_file)
    root = tree.getroot()
    
    # Find the route files and replace them with the new one
    input_elem = root.find(".//input")
    route_files_elem = input_elem.findall("route-files")
    
    # Remove existing route files
    for route_files in route_files_elem:
        input_elem.remove(route_files)
    
    # Create a new route-files element with the new route file
    new_route_files_elem = ET.SubElement(input_elem, "route-files")
    new_route_files_elem.set("value", new_route_file)
    
    tree.write(output_config_file)

def train(
    original_config_file,
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
    output_dir=os.path.join(os.path.dirname(__file__), "../output"),
    mixed_route_interval_length=300  # Add this parameter with a default value
):
    """
    Train a Shared Noisy Double DQN agent with mixed traffic levels in each episode.
    Generates a new route file for every episode with random mix of traffic levels.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "model"), exist_ok=True)
    
    # Generate the initial mixed traffic route file
    initial_route_file = "route_mixed_initial.rou.xml"
    mixed_config_file = "mixed_traffic.sumocfg"
    
    generate_mixed_traffic_route_file(original_config_file, initial_route_file, max_steps)
    update_config_with_route(original_config_file, initial_route_file, mixed_config_file)
    
    # Initialize the environment with the mixed traffic route file
    env = SumoEnvironment(
        config_file=mixed_config_file,
        use_gui=use_gui,
        num_seconds=max_steps,
        delta_time=5
    )
    
    intersection_groups = create_intersection_groups(env.intersection_types)
    state_dims, action_dims = create_state_action_dims(env)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = SharedNoisyDoubleDQNAgent(
        state_dims=state_dims,
        action_dims=action_dims,
        intersection_groups=intersection_groups,
        learning_rate=learning_rate,
        gamma=gamma,
        initial_sigma=1.0,
        sigma_decay=0.995,
        sigma_min=0.01,
        target_update=target_update,
        device=device
    )
    
    replay_buffers = {
        group_id: ReplayBuffer(buffer_size) for group_id in intersection_groups
    }
    
    metrics = {
        "episode_rewards": [],
        "episode_lengths": [],
        "average_queue_lengths": [],
        "average_waiting_times": [],
        "average_speeds": [],
        "training_times": [],
        "eval_rewards": [],
        "eval_queue_lengths": [],
        "eval_waiting_times": [],
        "eval_speeds": []
    }
    
    WARMUP_STEPS = 5000
    
    for episode in range(num_episodes):
        print(f"Episode {episode + 1}/{num_episodes}")
        
        # Generate a new mixed traffic route file for this episode
        route_file = f"route_mixed_episode_{episode+1}.rou.xml"
        generate_mixed_traffic_route_file(original_config_file, route_file, max_steps)
        update_config_with_route(original_config_file, route_file, mixed_config_file)
        
        states = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_queue_lengths = []
        episode_waiting_times = []
        episode_speeds = []
        
        start_time = time.time()
        done = False
        
        while not done:
            actions = {}
            for tl_id in env.traffic_lights:
                group_id = get_intersection_group(tl_id, intersection_groups)
                action = agent.select_action(tl_id, states[tl_id], group_id)
                actions[tl_id] = action
            
            next_states, rewards, dones, info = env.step(actions)
            done = any(dones.values())
            
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
            step_reward = sum(rewards.values()) / len(rewards)
            episode_reward += step_reward
            episode_length += 1
            
            queue_length = 0
            waiting_time = 0
            speed_sum = 0
            lane_count = 0
            
            for tl_id in env.traffic_lights:
                for lane in env.incoming_lanes[tl_id]:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                    queue_length += q
                    n_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    if n_vehicles > 0:
                        waiting_time += traci.lane.getWaitingTime(lane) / n_vehicles
                    speed_sum += traci.lane.getLastStepMeanSpeed(lane)
                    lane_count += 1
            
            avg_queue_length = queue_length / len(env.traffic_lights) if lane_count > 0 else 0
            avg_waiting_time = waiting_time / lane_count if lane_count > 0 else 0
            avg_speed = speed_sum / lane_count if lane_count > 0 else 0
            
            episode_queue_lengths.append(avg_queue_length)
            episode_waiting_times.append(avg_waiting_time)
            episode_speeds.append(avg_speed)
            
            for group_id in intersection_groups:
                if len(replay_buffers[group_id]) > WARMUP_STEPS:
                    agent.update(group_id, replay_buffers[group_id], batch_size, episode_reward=None)
            
            if episode_length >= max_steps:
                done = True
        
        end_time = time.time()
        training_time = end_time - start_time
        
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_lengths"].append(episode_length)
        metrics["average_queue_lengths"].append(np.mean(episode_queue_lengths))
        metrics["average_waiting_times"].append(np.mean(episode_waiting_times))
        metrics["average_speeds"].append(np.mean(episode_speeds))
        metrics["training_times"].append(training_time)
        
        print(f" Reward: {episode_reward:.2f}")
        print(f" Length: {episode_length}")
        print(f" Avg Queue: {np.mean(episode_queue_lengths):.2f}")
        print(f" Avg Wait : {np.mean(episode_waiting_times):.2f}")
        print(f" Avg Speed: {np.mean(episode_speeds):.2f}")
        print(f" Training Time: {training_time:.2f}s")
        
        for group_id in intersection_groups:
            agent.update(group_id, replay_buffers[group_id], batch_size,
                        episode_reward=episode_reward, episode_num=episode + 1)
        
        any_gid = list(intersection_groups.keys())[0]
        print(f" [Episode {episode + 1}] Total EpReward: {episode_reward:.2f} | Sigma: {agent.agents[any_gid].sigma:.4f}")
        
        if (episode + 1) % eval_interval == 0:
            eval_metrics = evaluate(env, agent, intersection_groups)
            metrics["eval_rewards"].append(eval_metrics["reward"])
            metrics["eval_queue_lengths"].append(eval_metrics["queue_length"])
            metrics["eval_waiting_times"].append(eval_metrics["waiting_time"])
            metrics["eval_speeds"].append(eval_metrics["speed"])
            
            print(" Evaluation:")
            print(f" Reward: {eval_metrics['reward']:.2f}")
            print(f" Avg Queue: {eval_metrics['queue_length']:.2f}")
            print(f" Avg Wait : {eval_metrics['waiting_time']:.2f}")
            print(f" Avg Speed: {eval_metrics['speed']:.2f}")
        
        if (episode + 1) % save_interval == 0:
            agent.save_models(f"{output_dir}/model/models_episode_{episode + 1}")
            save_metrics(metrics, f"{output_dir}/metrics.json")
            plot_metrics(metrics, output_dir)
        
        # Clean up old route files to save disk space
        if episode > 0 and os.path.exists(f"route_mixed_episode_{episode}.rou.xml"):
            os.remove(f"route_mixed_episode_{episode}.rou.xml")
        
        # Also clean up the initial route file after the first episode
        if episode == 0 and os.path.exists(initial_route_file):
            os.remove(initial_route_file)
    
    env.close()
    
    # Save final models and metrics
    agent.save_models(f"{output_dir}/models_final")
    save_metrics(metrics, f"{output_dir}/metrics.json")
    plot_metrics(metrics, output_dir)

def evaluate(env, agent, intersection_groups, num_episodes=3):
    """Evaluation function."""
    total_reward = 0.0
    total_queue_length = 0.0
    total_waiting_time = 0.0
    total_speed = 0.0
    
    for _ in range(num_episodes):
        states = env.reset()
        ep_reward = 0.0
        ep_queue_lengths = []
        ep_waiting_times = []
        ep_speeds = []
        done = False
        
        while not done:
            actions = {}
            for tl_id in env.traffic_lights:
                group_id = get_intersection_group(tl_id, intersection_groups)
                action = agent.select_action(tl_id, states[tl_id], group_id)
                actions[tl_id] = action
            
            next_states, rewards, dones, _ = env.step(actions)
            done = any(dones.values())
            states = next_states
            ep_reward += sum(rewards.values())
            
            queue_length = 0
            waiting_time = 0
            speed_sum = 0
            lane_count = 0
            
            for tl_id in env.traffic_lights:
                for lane in env.incoming_lanes[tl_id]:
                    q = traci.lane.getLastStepHaltingNumber(lane)
                    queue_length += q
                    n_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    if n_vehicles > 0:
                        waiting_time += traci.lane.getWaitingTime(lane) / n_vehicles
                    speed_sum += traci.lane.getLastStepMeanSpeed(lane)
                    lane_count += 1
            
            avg_queue_length = queue_length / len(env.traffic_lights) if lane_count > 0 else 0
            avg_waiting_time = waiting_time / lane_count if lane_count > 0 else 0
            avg_speed = speed_sum / lane_count if lane_count > 0 else 0
            
            ep_queue_lengths.append(avg_queue_length)
            ep_waiting_times.append(avg_waiting_time)
            ep_speeds.append(avg_speed)
        
        total_reward += ep_reward
        total_queue_length += np.mean(ep_queue_lengths)
        total_waiting_time += np.mean(ep_waiting_times)
        total_speed += np.mean(ep_speeds)
    
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
    """Plotting function."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    episodes = list(range(1, len(metrics["episode_rewards"]) + 1))
    eval_episodes = list(range(1, len(metrics["eval_rewards"]) + 1))
    
    plt.figure()
    plt.plot(episodes, metrics["episode_rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(fig_dir, "episode_rewards.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, metrics["average_queue_lengths"])
    plt.title("Average Queue Lengths")
    plt.xlabel("Episode")
    plt.ylabel("Queue Length")
    plt.savefig(os.path.join(fig_dir, "average_queue_lengths.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, metrics["average_waiting_times"])
    plt.title("Average Waiting Times")
    plt.xlabel("Episode")
    plt.ylabel("Waiting Time (s)")
    plt.savefig(os.path.join(fig_dir, "average_waiting_times.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, metrics["average_speeds"])
    plt.title("Average Speeds")
    plt.xlabel("Episode")
    plt.ylabel("Speed (m/s)")
    plt.savefig(os.path.join(fig_dir, "average_speeds.png"))
    plt.close()
    
    plt.figure()
    plt.plot(episodes, metrics["training_times"])
    plt.title("Training Times")
    plt.xlabel("Episode")
    plt.ylabel("Time (s)")
    plt.savefig(os.path.join(fig_dir, "training_times.png"))
    plt.close()
    
    if len(metrics["eval_rewards"]) > 0:
        eval_interval = len(episodes) // len(eval_episodes)
        real_eval_episodes = [eval_interval * i for i in range(1, len(eval_episodes) + 1)]
        
        plt.figure()
        plt.plot(real_eval_episodes, metrics["eval_rewards"])
        plt.title("Evaluation Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig(os.path.join(fig_dir, "eval_rewards.png"))
        plt.close()
        
        plt.figure()
        plt.plot(real_eval_episodes, metrics["eval_queue_lengths"])
        plt.title("Evaluation Queue Lengths")
        plt.xlabel("Episode")
        plt.ylabel("Queue Length")
        plt.savefig(os.path.join(fig_dir, "eval_queue_lengths.png"))
        plt.close()
        
        plt.figure()
        plt.plot(real_eval_episodes, metrics["eval_waiting_times"])
        plt.title("Evaluation Waiting Times")
        plt.xlabel("Episode")
        plt.ylabel("Waiting Time (s)")
        plt.savefig(os.path.join(fig_dir, "eval_waiting_times.png"))
        plt.close()
        
        plt.figure()
        plt.plot(real_eval_episodes, metrics["eval_speeds"])
        plt.title("Evaluation Speeds")
        plt.xlabel("Episode")
        plt.ylabel("Speed (m/s)")
        plt.savefig(os.path.join(fig_dir, "eval_speeds.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Noisy Double DQN agent with mixed traffic levels")
    default_config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../run.sumocfg"))
    parser.add_argument("--config_file", type=str, default=default_config_path, help="Path to the original SUMO config file")
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
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.path.dirname(__file__), "../output_mixed"), help="Directory to save models and results")
    parser.add_argument("--interval_length", type=int, default=300, help="Interval length in seconds for varying traffic (e.g., 300 for 5 mins)")

    args = parser.parse_args()

    # Make sure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    train(
        original_config_file=args.config_file,
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
        output_dir=args.output_dir,
        mixed_route_interval_length=args.interval_length
    )
