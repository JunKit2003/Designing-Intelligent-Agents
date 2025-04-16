import os
import sys
import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
import traci
import sumolib
import subprocess

# Add SUMO_HOME to path if not already there
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Set LIBSUMO as TRACI
os.environ["LIBSUMO_AS_TRACI"] = "1"

class SumoEnvironment:
    """
    SUMO Environment for reinforcement learning with traffic signal control
    """
    def __init__(
        self,
        config_file,
        use_gui=False,
        num_seconds=3600,
        max_depart_delay=100000,
        time_to_teleport=300,
        delta_time=2,
        yellow_time=2,
        min_green=5,
        max_green=50,
        reward_fn=None,
        alpha=0.7,
        beta=0.3,
        port=8813
    ):
        self.config_file = config_file
        self.use_gui = use_gui
        self.num_seconds = num_seconds
        self.max_depart_delay = max_depart_delay
        self.time_to_teleport = time_to_teleport
        self.delta_time = delta_time  # how many seconds each step represents
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.max_green = max_green
        #self.reward_fn = reward_fn if reward_fn else self._pressure_reward #currently using hybrid reward // Full Pressure based reward
        self.alpha = alpha
        self.beta = beta
        #self.reward_fn = reward_fn if reward_fn else lambda tl_id: self._hybrid_reward(tl_id) // Hybrid reward
        self.reward_fn = reward_fn if reward_fn else lambda tl_id: self._shaped_reward(tl_id)
        self.port = port

            
        # Ensure the ../output folder exists (so SUMO can write logs)
        os.makedirs("../output", exist_ok=True)
        
        # SUMO command to start the simulation
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", config_file,
            "--no-step-log", "true",
            "--waiting-time-memory", str(num_seconds),
            "--time-to-teleport", str(time_to_teleport),
            "--max-depart-delay", str(max_depart_delay),
            "--log", "../output/sumo.log",
            "--error-log", "../output/sumo_errors.log",
            "--no-warnings", "true",
            "--message-log", "../output/sumo_messages.log"
        ]

        
        # Start SUMO with TraCI
        traci.start(sumo_cmd)

        # Initialize traffic light data
        self._init_traffic_lights()
        
        # Store intersection types and shapes
        self.intersection_types = self._classify_intersections()
        
        # Current simulation step
        self.sim_step = 0
        
        # Traffic light phases
        self.phases = self._get_tl_phases()
        
        # Current traffic light states
        self.traffic_light_states = {}
        for tl_id in self.traffic_lights:
            self.traffic_light_states[tl_id] = {
                'current_phase': 0,
                'time_since_last_change': 0,
                'yellow_phase': False
            }
    
    def _init_traffic_lights(self):
        """Initialize traffic light data"""
        self.traffic_lights = list(traci.trafficlight.getIDList())
        self.traffic_light_node_ids = {}
        
        # Get mapping between traffic lights and nodes
        net = sumolib.net.readNet(self._get_net_file_path())
        for tl_id in self.traffic_lights:
            for node in net.getNodes():
                if node.getType() == "traffic_light" and node.getID() == tl_id:
                    self.traffic_light_node_ids[tl_id] = node.getID()
                    break
        
        # Get incoming and outgoing lanes for each traffic light
        self.incoming_lanes = {}
        self.outgoing_lanes = {}
        for tl_id in self.traffic_lights:
            self.incoming_lanes[tl_id] = list(traci.trafficlight.getControlledLanes(tl_id))
            self.outgoing_lanes[tl_id] = []
            for incoming in self.incoming_lanes[tl_id]:
                for outgoing in traci.lane.getLinks(incoming):
                    if outgoing[0] not in self.outgoing_lanes[tl_id]:
                        self.outgoing_lanes[tl_id].append(outgoing[0])
    
    def _get_net_file_path(self):
        """Extract the network file path from the config file"""
        tree = ET.parse(self.config_file)
        root = tree.getroot()
        for input_element in root.findall('.//input'):
            for net_file in input_element.findall('.//net-file'):
                return os.path.join(os.path.dirname(self.config_file), net_file.get('value'))
        return None
    
    def _get_tl_phases(self):
        """Get traffic light phases for each intersection"""
        phases = {}
        for tl_id in self.traffic_lights:
            phases[tl_id] = []
            logic = traci.trafficlight.getAllProgramLogics(tl_id)[0]
            for phase in logic.getPhases():
                phases[tl_id].append(phase.state)
        return phases
    
    def _classify_intersections(self):
        """
        Classify intersections based on their shape and structure
        Returns a dictionary mapping intersection types to lists of traffic light IDs
        """
        intersection_types = defaultdict(list)
        
        for tl_id in self.traffic_lights:
            # Features to classify the intersection
            num_incoming_lanes = len(self.incoming_lanes[tl_id])
            num_outgoing_lanes = len(self.outgoing_lanes[tl_id])
            num_phases = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].getPhases())
            
            # Create a simple intersection type signature
            type_signature = f"{num_incoming_lanes}_{num_outgoing_lanes}_{num_phases}"
            intersection_types[type_signature].append(tl_id)
            
        return dict(intersection_types)
    
    def get_state(self, tl_id):
        state = []

        # --- Existing State Features ---
        for lane in self.incoming_lanes[tl_id]:
            # Normalize queue (assume max 20 cars), waiting time (max 3600s)
            queue = traci.lane.getLastStepHaltingNumber(lane) / 20.0
            wait = traci.lane.getWaitingTime(lane) / 3600.0
            state.append(queue)
            state.append(wait)

        # Current phase (one-hot encoded)
        phase_id = self.traffic_light_states[tl_id]['current_phase']
        phase_one_hot = [0] * len(self.phases[tl_id])
        if phase_id < len(phase_one_hot): # Safety check for valid phase_id
            phase_one_hot[phase_id] = 1
        state.extend(phase_one_hot)

        # Time since last change (assume max 100s)
        time_since_change = self.traffic_light_states[tl_id]['time_since_last_change']
        state.append(min(time_since_change, 100) / 100.0)
        # --- End Existing State Features ---


        # --- Weighted Pressure Calculation (Single Value for Intersection) ---
        # Calculate total weighted incoming vehicles
        total_weighted_incoming = 0.0
        for lane in self.incoming_lanes[tl_id]:
            vehicles = traci.lane.getLastStepVehicleNumber(lane)
            try:
                # Use max speed as a proxy for lane weight/importance/capacity
                # Higher speed limit lanes contribute more to the pressure calculation
                weight = traci.lane.getMaxSpeed(lane)
                if weight <= 0: weight = 1.0 # Ensure weight is positive, default to 1 if speed is 0 or less
            except traci.exceptions.TraCIException:
                # Handle cases where lane info might be unavailable (e.g., internal lanes?)
                weight = 1.0 # Default weight if speed limit not available
            total_weighted_incoming += vehicles * weight

        # Calculate total weighted outgoing vehicles
        # Use the pre-calculated self.outgoing_lanes list from __init__
        total_weighted_outgoing = 0.0
        for out_lane in self.outgoing_lanes[tl_id]:
            # Check if the lane exists in the current simulation step/context (safety check)
            if out_lane in traci.lane.getIDList():
                vehicles = traci.lane.getLastStepVehicleNumber(out_lane)
                try:
                    weight = traci.lane.getMaxSpeed(out_lane)
                    if weight <= 0: weight = 1.0
                except traci.exceptions.TraCIException:
                    weight = 1.0 # Default weight
                total_weighted_outgoing += vehicles * weight
            # else: lane might be outside scope or internal - effectively skipped

        # Calculate raw weighted pressure: Weighted Incoming - Weighted Outgoing
        weighted_pressure = total_weighted_incoming - total_weighted_outgoing

        # Normalize the weighted pressure to roughly [-1, 1] range using tanh
        # The scaling_factor helps adjust the sensitivity of the tanh function.
        # You might need to TUNE this factor based on the typical range of 'weighted_pressure'
        # observed in your specific simulation scenario to ensure the output
        # utilizes the [-1, 1] range effectively and isn't always saturated at -1 or 1.
        # Start with a reasonable guess and adjust by observing the values.
        scaling_factor = 100.0 # Example scaling factor - ADJUST BASED ON OBSERVATION
        normalized_weighted_pressure = np.tanh(weighted_pressure / scaling_factor)

        # Append the single weighted and normalized pressure value to the state
        state.append(normalized_weighted_pressure)
        # --- End Weighted Pressure Calculation ---


        return np.array(state, dtype=np.float32)



    # def _hybrid_reward(self, tl_id, alpha=0.7, beta=0.3):
    #     """
    #     Hybrid reward: -alpha * pressure - beta * total waiting time on incoming lanes
    #     """
    #     # --- Pressure calculation ---
    #     incoming_pressure = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.incoming_lanes[tl_id])
    #     outgoing_pressure = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in self.outgoing_lanes[tl_id])
    #     pressure = incoming_pressure - outgoing_pressure

    #     # --- Waiting time calculation ---
    #     total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in self.incoming_lanes[tl_id])

    #     # --- Combined reward ---
    #     reward = (-alpha * pressure - beta * total_waiting_time)
    #     reward = reward / 100.0  # scale down
    #     reward = np.clip(reward, -1000, 0)

    #     return reward

    # def _pressure_reward(self, tl_id):
    #     """
    #     Pressure = |incoming - outgoing| at each intersection.
    #     Lower pressure = smoother flow.
    #     Returns a negative pressure to maximize in RL.
    #     """
    #     incoming_lanes = self.incoming_lanes[tl_id]
    #     incoming_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in incoming_lanes)

    #     outgoing_lanes_set = set()
    #     for lane in incoming_lanes:
    #         connections = traci.lane.getLinks(lane)
    #         for conn in connections:
    #             to_lane = conn[0]
    #             outgoing_lanes_set.add(to_lane)

    #     outgoing_count = sum(traci.lane.getLastStepVehicleNumber(lane) for lane in outgoing_lanes_set)

    #     pressure = abs(incoming_count - outgoing_count)
    #     return -pressure / 50 # reward is negative pressure


    def _shaped_reward(self, tl_id):
        """
        Reward function that combines:
        - Lower waiting time per vehicle
        - Lower queue length
        - Higher average speed
        """

        total_waiting_time = 0
        total_vehicles = 0
        total_queue = 0
        total_speed = 0
        num_speed_samples = 0

        for lane in self.incoming_lanes[tl_id]:
            num = traci.lane.getLastStepVehicleNumber(lane)
            wait = traci.lane.getWaitingTime(lane)
            queue = traci.lane.getLastStepHaltingNumber(lane)
            speed = traci.lane.getLastStepMeanSpeed(lane)

            total_waiting_time += wait
            total_vehicles += num
            total_queue += queue
            total_speed += speed
            num_speed_samples += 1

        # Avoid divide-by-zero errors
        avg_wait = total_waiting_time / (total_vehicles + 1e-6)
        avg_queue = total_queue / max(len(self.incoming_lanes[tl_id]), 1)
        avg_speed = total_speed / max(num_speed_samples, 1)

        # Reward: penalize wait & queue, reward speed
        reward = -0.4 * avg_wait - 0.3 * avg_queue + 0.3 * avg_speed

        # Scale to reasonable range
        reward = reward / 10.0
        reward = np.clip(reward, -100, 100)

        return reward




    # def _pressure_reward(self, tl_id): #Currently Testing Hybrid Reward
    #     """
    #     Calculate reward based on 'pressure'
    #     Pressure = -(incoming_vehicle_count - outgoing_vehicle_count)
    #     """
    #     incoming_pressure = 0
    #     for lane in self.incoming_lanes[tl_id]:
    #         incoming_pressure += traci.lane.getLastStepVehicleNumber(lane)
        
    #     outgoing_pressure = 0
    #     for lane in self.outgoing_lanes[tl_id]:
    #         outgoing_pressure += traci.lane.getLastStepVehicleNumber(lane)
        
    #     # Negative sign so that higher incoming vs. outgoing is penalized
    #     pressure = -(incoming_pressure - outgoing_pressure)
        
    #     return pressure
    
    def step(self, actions):
        """
        Execute one environment step by:
        1. Applying the given actions (set or change phases).
        2. Stepping the simulation delta_time times.
        3. Returning new states, rewards, done flags, and info.
        """
        # Apply actions for each traffic light
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)
        
        # Advance simulation for delta_time seconds
        for _ in range(self.delta_time):
            traci.simulationStep()
            self.sim_step += 1
            
            # Update traffic light timers
            for tl_id in self.traffic_lights:
                self.traffic_light_states[tl_id]['time_since_last_change'] += 1
        
        # Collect the next states and rewards
        states = {}
        rewards = {}
        for tl_id in self.traffic_lights:
            states[tl_id] = self.get_state(tl_id)
            rewards[tl_id] = self.reward_fn(tl_id)
        
        # Check if we've reached the end of the episode
        done = self.sim_step >= self.num_seconds
        dones = {tl_id: done for tl_id in self.traffic_lights}
        
        info = {
            'step': self.sim_step,
            'traffic_lights': self.traffic_lights,
            'intersection_types': self.intersection_types
        }
        
        return states, rewards, dones, info
    
    def _apply_action(self, tl_id, action):
        """Set or change the phase for the given traffic light ID."""
        tl_state = self.traffic_light_states[tl_id]
        
        # If currently in a yellow phase, check if we can switch to the new phase
        if tl_state['yellow_phase']:
            # If yellow time has elapsed, move on to the chosen phase
            if tl_state['time_since_last_change'] >= self.yellow_time:
                traci.trafficlight.setRedYellowGreenState(tl_id, self.phases[tl_id][action])
                tl_state['current_phase'] = action
                tl_state['yellow_phase'] = False
                tl_state['time_since_last_change'] = 0
        else:
            # If minimum green time has passed and the action is different from the current phase
            if (tl_state['time_since_last_change'] >= self.min_green 
                and action != tl_state['current_phase']):
                
                # Switch to a yellow phase first (for safety)
                current_phase_str = self.phases[tl_id][tl_state['current_phase']]
                yellow_state = ''.join(['y' if c.lower() == 'g' else 'r' for c in current_phase_str])
                
                traci.trafficlight.setRedYellowGreenState(tl_id, yellow_state)
                
                tl_state['yellow_phase'] = True
                tl_state['time_since_last_change'] = 0
    
    def reset(self):
        """
        Reset the environment to time=0.
        This closes the current connection and restarts SUMO from scratch.
        """
        # Close existing TraCI connection
        traci.close()
        
        # Ensure the ../output folder still exists
        os.makedirs("../output", exist_ok=True)
        
        # Start SUMO again with the same config (second sumo_cmd)
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        sumo_cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--no-step-log", "true",
            "--waiting-time-memory", str(self.num_seconds),
            "--time-to-teleport", str(self.time_to_teleport),
            "--max-depart-delay", str(self.max_depart_delay),
            "--log", "../output/sumo.log",
            "--error-log", "../output/sumo_errors.log",
            "--no-warnings", "true",
            "--message-log", "../output/sumo_messages.log"
        ]
        traci.start(sumo_cmd)

        # for tl_id in self.traffic_lights:
        #     print(f"[DEBUG] Before: {tl_id} has program", traci.trafficlight.getProgram(tl_id))
        #     traci.trafficlight.setProgram(tl_id, "0")
        #     print(f"[DEBUG] After: {tl_id} has program", traci.trafficlight.getProgram(tl_id))

        
        # Reset simulation step
        self.sim_step = 0
        
        # Reset each traffic light to its initial state and set initial green
        for tl_id in self.traffic_lights:
            self.traffic_light_states[tl_id] = {
                'current_phase': 0,
                'time_since_last_change': 0,
                'yellow_phase': False
            }
            traci.trafficlight.setRedYellowGreenState(tl_id, self.phases[tl_id][0])
        
        # Return initial states
        states = {}
        for tl_id in self.traffic_lights:
            states[tl_id] = self.get_state(tl_id)
        
        return states
    
    def close(self):
        """Close the environment and the underlying TraCI connection."""
        traci.close()
