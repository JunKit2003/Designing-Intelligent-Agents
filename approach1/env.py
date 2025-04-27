# env.py

import os
import time
import xml.etree.ElementTree as ET
import traci
import sumolib
import networkx as nx
import numpy as np
import math
import torch
from torch import nn

os.environ["LIBSUMO_AS_TRACI"] = "1"

class ATSCEnvironment:
    """
    Environment that constructs a hierarchical graph representation of the traffic network
    with adaptive vehicle prioritization at the movement level.
    
    This implementation uses a parameterized attention mechanism (Method 2) that allows
    different types of vehicles to have different priorities/weightage. The model can be 
    trained with just 1 type of vehicles, and during actual usage, we can simply tweak 
    the weightages of these vehicles to include virtual priority.
    """

    def __init__(self, sumocfg_file, sumo_binary="sumo", gui=False, simulation_step=1, 
                 segment_length=10, vehicle_types=None, vehicle_weights=None):
        self.sumocfg_file = sumocfg_file
        self.sumo_binary = sumo_binary + ("-gui" if gui else "")
        self.simulation_step = simulation_step
        self.segment_length = segment_length
        
        # Set up vehicle types and their weights
        self.vehicle_types = vehicle_types if vehicle_types else ["passenger"]
        
        # Default weights (equal for all types)
        default_weights = {vtype: 1.0 for vtype in self.vehicle_types}
        self.vehicle_weights = vehicle_weights if vehicle_weights else default_weights
        
        self.net_file = self._get_network_file_from_config()
        self.net = sumolib.net.readNet(self.net_file)

        self.tl_ids = []
        self.step_count = 0
        
        # Attention mechanism parameters
        self.attention_heads = 8
        self.attention_dropout = 0.1

    def _get_network_file_from_config(self):
        """Extract the network file path from the SUMO config file"""
        tree = ET.parse(self.sumocfg_file)
        root = tree.getroot()
        net_file = None
        for input_tag in root.findall('input'):
            net_file_elem = input_tag.find('net-file')
            if net_file_elem is not None:
                net_file = net_file_elem.attrib['value']
                break
        if net_file is None:
            raise ValueError("Network file not found in the .sumocfg file.")
        config_dir = os.path.dirname(self.sumocfg_file)
        return os.path.join(config_dir, net_file)

    def start(self):
        """Start the SUMO simulation"""
        sumo_cmd = [self.sumo_binary, "-c", self.sumocfg_file, "--start"]
        traci.start(sumo_cmd)
        self.tl_ids = traci.trafficlight.getIDList()
        print("SUMO started. Controlled TLs:", self.tl_ids)

    def close(self):
        """Close the SUMO simulation"""
        traci.close()
        print("SUMO closed.")

    def reset(self):
        """Reset the environment and return initial observation"""
        self.close()
        time.sleep(1)
        self.start()
        self.step_count = 0
        return self.get_observation()

    def step(self, actions):
        """
        Take actions in the environment
        
        Args:
            actions: dict { tl_id -> phase_index }
            
        Returns:
            observation, reward, done, info
        """
        for tl_id, phase_idx in actions.items():
            traci.trafficlight.setPhase(tl_id, phase_idx)

        for _ in range(self.simulation_step):
            traci.simulationStep()
            self.step_count += 1

        obs = self.get_observation()
        reward = self.compute_reward()
        done = traci.simulation.getMinExpectedNumber() <= 0
        info = {"step": self.step_count}
        return obs, reward, done, info
    
    def preprocess_graph_for_model(self, graph):
        """
        Preprocess NetworkX graph to make it compatible with the model.
        Adds necessary attributes that the model expects.
        """
        # Add required attributes to the graph
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') == 'intersection':
                # Store intersection indices for easy access
                if not hasattr(graph, 'intersection_nodes'):
                    graph.intersection_nodes = []
                graph.intersection_nodes.append(node)
        
        return graph


    def get_observation(self):
        """
        Build a hierarchical graph representation based on TransferLight encoding:
        - Segments: Divides lanes into equal-sized segments
        - Movements: Aggregates segments using weighted attention by vehicle type
        - Intersections: Top level for decision making
        
        Returns a networkx.DiGraph G with the hierarchical structure
        """
        G = nx.DiGraph()
        
        # Step 1: Create lane segment nodes
        segments_by_lane = {}  # Maps lane_id -> list of segment_ids
        segment_features = {}  # Maps segment_id -> features
        
        # Get all lanes controlled by traffic lights
        lane_set = set()
        for tl_id in self.tl_ids:
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            lane_set.update(controlled_lanes)
        
        lane_list = list(lane_set)
        
        # Create segments for each lane
        for lane_id in lane_list:
            lane_length = traci.lane.getLength(lane_id)
            num_segments = max(1, int(np.ceil(lane_length / self.segment_length)))
            segments_by_lane[lane_id] = []
            
            # Create segment nodes
            for i in range(num_segments):
                segment_id = f"{lane_id}_{i}"
                segments_by_lane[lane_id].append(segment_id)
                
                # Get segment boundaries
                start_pos = i * self.segment_length
                end_pos = min((i + 1) * self.segment_length, lane_length)
                
                # Extract vehicle distribution in this segment
                segment_vehicles = self._get_vehicles_in_segment(lane_id, start_pos, end_pos)
                vehicle_dist = self._get_vehicle_type_distribution(segment_vehicles)
                
                # Calculate positional encoding as in TransferLight
                segment_pos = i / num_segments  # Normalized position
                pos_encoding = np.sin(start_pos / lane_length * math.pi) + \
                               np.cos(end_pos / lane_length * math.pi)
                
                # Additional features
                is_green = self._check_lane_green(lane_id)
                avg_speed = traci.lane.getLastStepMeanSpeed(lane_id)
                
                # Calculate segment density (vehicles per meter)
                segment_density = len(segment_vehicles) / (end_pos - start_pos) if (end_pos - start_pos) > 0 else 0
                
                # Feature vector with vehicle type distribution
                feature_vector = []
                
                # Base features (density, position, signal state)
                feature_vector.extend([segment_density, pos_encoding, is_green, segment_pos, avg_speed])
                
                # Add vehicle type distribution features
                for vtype in self.vehicle_types:
                    type_density = vehicle_dist.get(vtype, 0) / (end_pos - start_pos) if (end_pos - start_pos) > 0 else 0
                    feature_vector.append(type_density)
                
                # Store the complete feature set
                segment_features[segment_id] = {
                    'feature_vector': np.array(feature_vector),
                    'vehicle_dist': vehicle_dist,
                    'position': segment_pos,
                    'is_green': is_green,
                    'density': segment_density,
                    'avg_speed': avg_speed
                }
                
                # Add node to graph with features
                G.add_node(segment_id, 
                           type='segment', 
                           features=np.array(feature_vector),
                           lane_id=lane_id,
                           segment_idx=i)
                
                # Connect segments within the same lane
                if i > 0:
                    prev_segment_id = f"{lane_id}_{i-1}"
                    G.add_edge(prev_segment_id, segment_id, edge_type='intra_lane')
        
        # Step 2: Define movements and create movement nodes
        movements = self._get_movement_definitions()
        
        for movement_id, (from_lane, to_lane) in movements.items():
            if from_lane not in segments_by_lane:
                continue
                
            from_segments = segments_by_lane[from_lane]
            
            # Skip movements with no segments
            if not from_segments:
                continue
                
            # Apply attention mechanism to aggregate segment features
            movement_features = self._apply_parameterized_attention(from_segments, segment_features)
            
            # Add movement node to graph
            G.add_node(movement_id, 
                      type='movement', 
                      features=movement_features,
                      from_lane=from_lane,
                      to_lane=to_lane)
            
            # Connect segments to this movement
            for segment_id in from_segments:
                G.add_edge(segment_id, movement_id, edge_type='segment_to_movement')
        
        # Step 3: Create intersection nodes and connect them to movements
        for tl_id in self.tl_ids:
            # Compute intersection features
            intersection_features = self._compute_intersection_features(tl_id)
            
            # Add intersection node
            G.add_node(tl_id, 
                      type='intersection',
                      features=intersection_features)
            
            # Connect movements to this intersection
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            for movement_id, (from_lane, _) in movements.items():
                if from_lane in controlled_lanes:
                    G.add_edge(movement_id, tl_id, edge_type='movement_to_intersection')
                    # Bidirectional connection to allow message passing in both directions
                    G.add_edge(tl_id, movement_id, edge_type='intersection_to_movement')
        
        # Step 4: Connect lanes based on connectivity in the road network
        for lane_id in lane_list:
            if lane_id not in segments_by_lane or not segments_by_lane[lane_id]:
                continue
                
            # Get outgoing connections
            conn_info = traci.lane.getLinks(lane_id)
            
            # Connect last segment of current lane to first segment of connected lanes
            last_segment = segments_by_lane[lane_id][-1]
            
            for c in conn_info:
                next_lane = c[0]
                if next_lane in segments_by_lane and segments_by_lane[next_lane]:
                    next_first_segment = segments_by_lane[next_lane][0]
                    G.add_edge(last_segment, next_first_segment, edge_type='inter_lane')
        
         # Preprocess the graph for the model
        G = self.preprocess_graph_for_model(G)
        
        return G

    def _get_vehicles_in_segment(self, lane_id, start_pos, end_pos):
        """
        Get all vehicles in a specific segment of a lane
        
        Args:
            lane_id: Lane identifier
            start_pos: Start position of segment (meters from lane start)
            end_pos: End position of segment (meters from lane start)
            
        Returns:
            List of vehicle IDs in the segment
        """
        vehicles = []
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
        
        for veh_id in vehicle_ids:
            lane_pos = traci.vehicle.getLanePosition(veh_id)
            if start_pos <= lane_pos < end_pos:
                vehicles.append(veh_id)
                
        return vehicles
    
    def _get_vehicle_type_distribution(self, vehicle_ids):
        """
        Get distribution of different vehicle types in a segment
        
        Args:
            vehicle_ids: List of vehicle IDs
            
        Returns:
            Dictionary mapping vehicle types to counts
        """
        type_counts = {vtype: 0 for vtype in self.vehicle_types}
        
        for veh_id in vehicle_ids:
            veh_type = traci.vehicle.getTypeID(veh_id)
            # Map to known type or default to first type
            if veh_type in type_counts:
                type_counts[veh_type] += 1
            elif self.vehicle_types:
                type_counts[self.vehicle_types[0]] += 1
                
        return type_counts
    
    def _apply_parameterized_attention(self, segment_ids, segment_features):
        """
        Key method for Method 2: Parameterized Attention Mechanism
        This aggregates segment features using attention weights based on vehicle types
        
        Args:
            segment_ids: List of segment IDs to aggregate
            segment_features: Dictionary mapping segment IDs to feature dictionaries
            
        Returns:
            Aggregated feature vector for the movement
        """
        if not segment_ids:
            # Return zero vector with proper dimension if no segments
            base_dim = 5  # density, pos_encoding, is_green, position, avg_speed
            total_dim = base_dim + len(self.vehicle_types)
            return np.zeros(total_dim)
        
        # Step 1: Extract vehicle type distributions from segments
        vehicle_type_matrices = []
        for segment_id in segment_ids:
            vehicle_dist = segment_features[segment_id]['vehicle_dist']
            # Convert to weighted distribution based on priority
            weighted_dist = np.array([
                vehicle_dist.get(vtype, 0) * self.vehicle_weights.get(vtype, 1.0)
                for vtype in self.vehicle_types
            ])
            vehicle_type_matrices.append(weighted_dist)
        
        vehicle_type_matrix = np.stack(vehicle_type_matrices)
        
        # Step 2: Calculate segment position factors (higher weight closer to intersection)
        position_factors = np.array([
            1.0 - segment_features[segment_id]['position']
            for segment_id in segment_ids
        ])
        
        # Step 3: Calculate attention weights using weighted vehicle counts and position
        weighted_counts = np.sum(vehicle_type_matrix, axis=1)  # Sum over vehicle types
        attention_scores = weighted_counts * position_factors
        
        # Step 4: Apply softmax to get normalized attention weights
        attention_scores = np.exp(attention_scores - np.max(attention_scores))  # Numerical stability
        attention_weights = attention_scores / (np.sum(attention_scores) + 1e-10)
        
        # Step 5: Weight and aggregate segment features
        aggregated_features = np.zeros_like(segment_features[segment_ids[0]]['feature_vector'])
        
        for i, segment_id in enumerate(segment_ids):
            aggregated_features += segment_features[segment_id]['feature_vector'] * attention_weights[i]
        
        return aggregated_features
    
    def _get_movement_definitions(self):
        """
        Extract movement definitions (from lane -> to lane pairs)
        
        Returns:
            Dictionary mapping movement IDs to (from_lane, to_lane) pairs
        """
        movements = {}
        
        for tl_id in self.tl_ids:
            # Get controlled links information
            link_data = traci.trafficlight.getControlledLinks(tl_id)
            
            for link_idx, links in enumerate(link_data):
                if not links:  # Skip empty links
                    continue
                    
                for i, link_tuple in enumerate(links):
                    from_lane, to_lane, _ = link_tuple
                    movement_id = f"{tl_id}_m{link_idx}_{i}"
                    movements[movement_id] = (from_lane, to_lane)
        
        return movements
    
    def _compute_intersection_features(self, tl_id):
        """
        Compute features for an intersection node
        
        Args:
            tl_id: Traffic light identifier
            
        Returns:
            Feature vector for the intersection
        """
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        
        # Calculate pressure for controlled lanes
        total_pressure = 0
        total_queue = 0
        weighted_pressure = 0
        lane_count = len(controlled_lanes) if controlled_lanes else 1
        
        for lane_id in controlled_lanes:
            # Get vehicles on this lane
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
            vehicle_dist = self._get_vehicle_type_distribution(vehicle_ids)
            
            # Compute pressure (incoming vs outgoing vehicles)
            incoming_count = len(vehicle_ids)
            
            # Get outgoing lanes and their vehicles
            outgoing_lanes = self._get_outgoing_lanes(lane_id)
            outgoing_count = sum(len(traci.lane.getLastStepVehicleIDs(out_lane)) 
                               for out_lane in outgoing_lanes)
            
            # Basic pressure
            lane_pressure = incoming_count - outgoing_count
            total_pressure += lane_pressure
            
            # Weighted pressure (based on vehicle type priorities)
            lane_weighted_pressure = sum(count * self.vehicle_weights.get(vtype, 1.0) 
                                      for vtype, count in vehicle_dist.items())
            lane_weighted_pressure -= outgoing_count  # Simple approximation for outgoing
            weighted_pressure += lane_weighted_pressure
            
            # Queue count
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
        
        avg_pressure = total_pressure / lane_count
        avg_queue = total_queue / lane_count
        avg_weighted_pressure = weighted_pressure / lane_count
        
        # Get current phase index
        current_phase = traci.trafficlight.getPhase(tl_id)
        phase_count = len(traci.trafficlight.getAllProgramLogics(tl_id)[0].phases)
        normalized_phase = current_phase / max(1, phase_count - 1) if phase_count > 0 else 0
        
        # Return feature vector
        return np.array([
            avg_pressure,
            avg_weighted_pressure,
            avg_queue,
            normalized_phase
        ])
    
    def _get_outgoing_lanes(self, lane_id):
        """
        Get the outgoing lanes connected to the given lane
        
        Args:
            lane_id: Lane identifier
            
        Returns:
            List of outgoing lane IDs
        """
        outgoing_lanes = []
        conn_info = traci.lane.getLinks(lane_id)
        
        for c in conn_info:
            if c:  # Skip empty connections
                outgoing_lanes.append(c[0])  # c[0] is the to-lane ID
                
        return outgoing_lanes

    def _check_lane_green(self, lane_id):
        """
        Check if a lane has a green light.
        """
        for tl_id in self.tl_ids:
            if lane_id in traci.trafficlight.getControlledLanes(tl_id):
                # Get traffic light state
                state_string = traci.trafficlight.getRedYellowGreenState(tl_id)
                link_info = traci.trafficlight.getControlledLinks(tl_id)
                
                # Loop through all connections
                for link_idx, connections in enumerate(link_info):
                    if not connections:
                        continue
                    
                    for conn in connections:
                        # Check if this connection is for our lane
                        if conn[0] == lane_id:
                            # Use the link_idx directly instead of trying to parse conn[2]
                            if 0 <= link_idx < len(state_string):
                                current_state = state_string[link_idx]
                                if current_state == 'G' or current_state == 'g':
                                    return 1.0
                return 0.0
        return 0.0



    def compute_reward(self):
        """
        Implement the log-distance reward from TransferLight paper with vehicle type weighting.
        This rewards prioritizes vehicles near intersections while remaining bounded.
        """
        total_reward = 0
        
        for tl_id in self.tl_ids:
            intersection_reward = 0
            controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
            
            for lane_id in controlled_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane_id)
                lane_length = traci.lane.getLength(lane_id)
                
                for veh_id in vehicle_ids:
                    # Get vehicle type and its weight
                    veh_type = traci.vehicle.getTypeID(veh_id)
                    weight = self.vehicle_weights.get(veh_type, 1.0)
                    
                    # Get distance from intersection (= lane length - vehicle position)
                    veh_pos = traci.vehicle.getLanePosition(veh_id)
                    distance = lane_length - veh_pos
                    
                    # Log-distance reward: higher reward for vehicles closer to intersection
                    # Apply vehicle type weight to prioritize certain types
                    vehicle_reward = -weight * np.log(1 + distance)
                    intersection_reward += vehicle_reward
            
            total_reward += intersection_reward
            
        return total_reward
    
    def set_vehicle_weights(self, vehicle_weights):
        """
        Update the weights for different vehicle types.
        This can be called at runtime to adjust priorities without retraining.
        
        Args:
            vehicle_weights: Dictionary mapping vehicle types to weight values
        """
        # Validate vehicle types
        for vtype in vehicle_weights:
            if vtype not in self.vehicle_types:
                print(f"Adding new vehicle type '{vtype}' to known types.")
                self.vehicle_types.append(vtype)
        
        # Update weights
        self.vehicle_weights.update(vehicle_weights)
        print(f"Updated vehicle weights: {self.vehicle_weights}")


