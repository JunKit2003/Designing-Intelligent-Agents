import traci
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sumo_env import SumoEnvironment

def run_single_baseline(config_file, max_steps=3600, use_gui=False):
    env = SumoEnvironment(
        config_file=config_file,
        use_gui=use_gui,
        num_seconds=max_steps
    )

    queue_lengths = []
    waiting_times = []
    speeds = []

    done = False
    while not done:
        traci.simulationStep()
        env.sim_step += 1

        total_queue = 0
        total_wait = 0
        total_speed = 0

        for tl_id in env.traffic_lights:
            for lane in env.incoming_lanes[tl_id]:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
                total_wait += traci.lane.getWaitingTime(lane)
                total_speed += traci.lane.getLastStepMeanSpeed(lane)

        avg_queue = total_queue / len(env.traffic_lights)
        avg_wait = total_wait / len(env.traffic_lights)
        avg_speed = total_speed / (len(env.traffic_lights) * len(env.incoming_lanes[env.traffic_lights[0]]))

        queue_lengths.append(avg_queue)
        waiting_times.append(avg_wait)
        speeds.append(avg_speed)

        done = env.sim_step >= max_steps

    env.close()
    return queue_lengths, waiting_times, speeds


def run_baseline_multiple(config_file, max_steps=3600, runs=100, use_gui=False, output_dir="./acosta/output_no_rl"):
    print(f"Running baseline (no RL) for {runs} episodes...\n")

    # Per-episode averages
    avg_queues_per_episode = []
    avg_waits_per_episode = []
    avg_speeds_per_episode = []

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    for i in range(runs):
        print(f"▶️ Run {i+1}/{runs}")
        q, w, s = run_single_baseline(config_file, max_steps=max_steps, use_gui=use_gui)

        avg_queues_per_episode.append(np.mean(q))
        avg_waits_per_episode.append(np.mean(w))
        avg_speeds_per_episode.append(np.mean(s))

        episodes = list(range(1, len(avg_queues_per_episode) + 1))

        # Plot updated metrics after each episode
        plt.figure()
        plt.plot(episodes, avg_queues_per_episode)
        plt.title("Avg Queue Length Per Episode (No RL)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Queue Length")
        plt.savefig(os.path.join(fig_dir, "queue_length_no_rl.png"))
        plt.close()

        plt.figure()
        plt.plot(episodes, avg_waits_per_episode)
        plt.title("Avg Waiting Time Per Episode (No RL)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Waiting Time")
        plt.savefig(os.path.join(fig_dir, "waiting_time_no_rl.png"))
        plt.close()

        plt.figure()
        plt.plot(episodes, avg_speeds_per_episode)
        plt.title("Avg Speed Per Episode (No RL)")
        plt.xlabel("Episode")
        plt.ylabel("Avg Speed (m/s)")
        plt.savefig(os.path.join(fig_dir, "speed_no_rl.png"))
        plt.close()

    # Final summary
    print("\n📊 Baseline Summary Over 100 Runs:")
    print(f"  Final Avg Queue Length: {np.mean(avg_queues_per_episode):.2f}")
    print(f"  Final Avg Waiting Time: {np.mean(avg_waits_per_episode):.2f}")
    print(f"  Final Avg Speed: {np.mean(avg_speeds_per_episode):.2f} m/s")


if __name__ == "__main__":
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../run.sumocfg"))
    run_baseline_multiple(config_file=config_path, max_steps=3600, runs=100, use_gui=False)
