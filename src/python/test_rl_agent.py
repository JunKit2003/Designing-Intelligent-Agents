import traci
import os
import sys
import pickle
import numpy as np

# Load trained Q-table
with open("python/q_table.pkl", "rb") as f:
    Q_TABLE = pickle.load(f)

# Start SUMO
SUMO_BINARY = "sumo-gui"
SUMO_CONFIG = "config/GeorgeTown.sumo.cfg"
traci.start([SUMO_BINARY, "-c", SUMO_CONFIG])

traffic_lights = traci.trafficlight.getIDList()

def get_state(tl):
    """ Returns a state based on queue length at intersection """
    lane_ids = traci.trafficlight.getControlledLanes(tl)
    queue_length = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids])
    return min(queue_length // 5, 3)

# Run simulation with trained agent
for step in range(500):
    traci.simulationStep()

    for tl in traffic_lights:
        state = get_state(tl)
        action = np.argmax(Q_TABLE[tl][state])  # Use learned policy

        if action == 1:  # Switch phase
            traci.trafficlight.setPhase(tl, (traci.trafficlight.getPhase(tl) + 1) % 4)

traci.close()
