# generate_curriculum.py

import os
import shutil
import subprocess
import random
import math
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET

class CurriculumGenerator:
    def __init__(self, sumo_home=None, output_dir="curriculum_scenarios"):
        """Initialize the curriculum generator."""
        if sumo_home:
            os.environ["SUMO_HOME"] = sumo_home
        elif "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please set SUMO_HOME environment variable or provide it as argument")
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define traffic parameters for different levels
        self.traffic_levels = {
            # "light": {
            #     "period": 3.0,            # High period = low frequency
            #     "fringe_factor": 5.0,     # Moderate through-traffic
            #     "depart_speed": "random", # Random departure speed
            #     "min_distance": 300,      # Minimum trip distance (meters)
            #     "max_distance": 2000,     # Maximum trip distance (meters)
            #     "route_factor": 1.3       # Route randomization factor
            # },
            "medium": {
                "period": 1.5,
                "fringe_factor": 10.0,
                "depart_speed": "random",
                "min_distance": 300,
                "max_distance": 3000,
                "route_factor": 1.5
            },
            "heavy": {
                "period": 0.8,
                "fringe_factor": 15.0,
                "depart_speed": "random",
                "min_distance": 200,
                "max_distance": 4000,
                "route_factor": 1.8
            }
        }
        
        # Default simulation parameters
        self.sim_begin = 0
        self.sim_end = 3600  # 1 hour simulation
        
    def _run_command(self, command):
        """Run a shell command and return the output."""
        print(f"Running: {' '.join(command)}")
        return subprocess.run(command, check=True, text=True, capture_output=True)
    
    def _create_node_file(self, nodes, scenario_dir, scenario_name):
        """Create a node file for SUMO network."""
        node_file = os.path.join(scenario_dir, f"{scenario_name}.nod.xml")
        
        root = ET.Element("nodes")
        for node in nodes:
            ET.SubElement(root, "node", node)
        
        tree = ET.ElementTree(root)
        tree.write(node_file, encoding="utf-8", xml_declaration=True)
        
        return node_file
    
    def _create_edge_file(self, edges, scenario_dir, scenario_name):
        """Create an edge file for SUMO network."""
        edge_file = os.path.join(scenario_dir, f"{scenario_name}.edg.xml")
        
        root = ET.Element("edges")
        for edge in edges:
            ET.SubElement(root, "edge", edge)
        
        tree = ET.ElementTree(root)
        tree.write(edge_file, encoding="utf-8", xml_declaration=True)
        
        return edge_file
    
    def _generate_traffic(self, net_file, output_file, traffic_level, seed=42):
        """Generate traffic for a network using randomTrips.py with improved error handling."""
        # Get traffic parameters for the specified level
        params = self.traffic_levels[traffic_level]
        
        # Create a unique prefix for each traffic level to avoid ID conflicts
        vehicle_prefix = f"{traffic_level}_veh"
        
        # Build randomTrips.py command
        random_trips_script = os.path.join(os.environ["SUMO_HOME"], "tools", "randomTrips.py")
        
        # Generate trips file
        trips_file = output_file.replace(".rou.xml", ".trips.xml")
        
        try:
            # First attempt with all parameters
            cmd = [
                "python", random_trips_script,
                "-n", net_file,
                "-o", trips_file,
                "-b", str(self.sim_begin),
                "-e", str(self.sim_end),
                "-p", str(params["period"]),
                "--fringe-factor", str(params["fringe_factor"]),
                "--min-distance", str(params["min_distance"]),
                "--max-distance", str(params["max_distance"]),
                "--seed", str(seed),
                "--prefix", vehicle_prefix
            ]
            self._run_command(cmd)
            
            # Convert trips to routes with error handling
            duarouter_cmd = [
                "duarouter", 
                "--route-files", trips_file,
                "--net-file", net_file,
                "--output-file", output_file,
                "--ignore-errors", "true",
                "--no-warnings", "true"
            ]
            self._run_command(duarouter_cmd)
            
            return output_file
        except Exception as e:
            print(f"Error generating traffic: {e}")
            print("Creating minimal valid route file as fallback")
            self._create_minimal_route_file(output_file, net_file, vehicle_prefix)
            return output_file

    def _create_minimal_route_file(self, route_file, net_file, vehicle_prefix):
        """Create a minimal valid route file as fallback."""
        # Get a list of edge IDs from the network
        import sumolib
        net = sumolib.net.readNet(net_file)
        edges = [edge.getID() for edge in net.getEdges()]
        
        # Find potential routes (pairs of edges that could form valid routes)
        potential_routes = []
        for from_edge in edges:
            for to_edge in edges:
                if from_edge != to_edge and not (from_edge.endswith("_to_center") and to_edge.endswith("_to_center")):
                    potential_routes.append((from_edge, to_edge))
        
        # If no potential routes, create simple placeholder
        if not potential_routes:
            potential_routes = [(edges[0], edges[-1])] if len(edges) > 1 else [(edges[0], edges[0])]
        
        # Write minimal route file
        with open(route_file, 'w') as f:
            f.write("""<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="passenger" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="70" guiShape="passenger"/>
""")
            
            # Add a few vehicles with the valid routes
            for i, (from_edge, to_edge) in enumerate(potential_routes[:5]):
                f.write(f"""    <vehicle id="{vehicle_prefix}_{i}" type="passenger" depart="{i*10}" color="1,0,0">
        <route edges="{from_edge} {to_edge}"/>
    </vehicle>
""")
            
            f.write("</routes>")

    def _create_config_file(self, net_file, route_file, scenario_dir, name, traffic_level):
        """Create a SUMO configuration file for a specific traffic level."""
        # Create subdirectory for this traffic level
        traffic_dir = os.path.join(scenario_dir, traffic_level)
        os.makedirs(traffic_dir, exist_ok=True)
        
        # Copy network file to traffic directory
        net_file_copy = os.path.join(traffic_dir, os.path.basename(net_file))
        shutil.copy(net_file, net_file_copy)
        
        # Copy route file to traffic directory
        route_file_copy = os.path.join(traffic_dir, os.path.basename(route_file))
        shutil.copy(route_file, route_file_copy)
        
        # Create config file in traffic directory
        config_path = os.path.join(traffic_dir, f"{name}_{traffic_level}.sumocfg")
        
        with open(config_path, "w") as config_file:
            config_file.write(
                f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.basename(net_file_copy)}"/>
        <route-files value="{os.path.basename(route_file_copy)}"/>
    </input>
    <time>
        <begin value="{self.sim_begin}"/>
        <end value="{self.sim_end}"/>
    </time>
    <processing>
        <time-to-teleport value="300"/>
        <ignore-junction-blocker value="30"/>
        <collision.action value="warn"/>
        <collision.stoptime value="5"/>
    </processing>
    <report>
        <verbose value="true"/>
        <duration-log.statistics value="true"/>
        <no-step-log value="true"/>
    </report>
</configuration>
"""
            )
        
        return config_path
    
    def create_t_junction(self, num_lanes=1, scenario_name=None):
        """Create a T-junction intersection with specified number of lanes."""
        if scenario_name is None:
            scenario_name = f"t_junction_{num_lanes}lane"
        
        scenario_dir = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Define nodes
        nodes = [
            {"id": "center", "x": "0", "y": "0", "type": "traffic_light"},
            {"id": "east", "x": "200", "y": "0", "type": "priority"},
            {"id": "north", "x": "0", "y": "200", "type": "priority"},
            {"id": "west", "x": "-200", "y": "0", "type": "priority"}
        ]
        node_file = self._create_node_file(nodes, scenario_dir, scenario_name)
        
        # Define edges
        edges = [
            {"id": "east_to_center", "from": "east", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_east", "from": "center", "to": "east", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "north_to_center", "from": "north", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_north", "from": "center", "to": "north", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "west_to_center", "from": "west", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_west", "from": "center", "to": "west", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"}
        ]
        edge_file = self._create_edge_file(edges, scenario_dir, scenario_name)
        
        # Create network file
        net_file = os.path.join(scenario_dir, f"{scenario_name}.net.xml")
        cmd = [
            "netconvert",
            "--node-files", node_file,
            "--edge-files", edge_file,
            "--output", net_file,
            "--no-turnarounds", "true",
            "--tls.green.time", "30",
            "--tls.yellow.time", "4",
            "--tls.red.time", "1"
        ]
        self._run_command(cmd)
        
        # Generate traffic for different levels in separate directories
        for level in self.traffic_levels.keys():
            level_dir = os.path.join(scenario_dir, level)
            os.makedirs(level_dir, exist_ok=True)
            
            route_file = os.path.join(scenario_dir, f"{scenario_name}_{level}.rou.xml")
            self._generate_traffic(net_file, route_file, level)
            
            # Create separate config file for each traffic level
            self._create_config_file(net_file, route_file, scenario_dir, scenario_name, level)
        
        return scenario_dir
    
    def create_four_way_junction(self, num_lanes=1, scenario_name=None):
        """Create a four-way intersection with specified number of lanes."""
        if scenario_name is None:
            scenario_name = f"four_way_{num_lanes}lane"
        
        scenario_dir = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Define nodes
        nodes = [
            {"id": "center", "x": "0", "y": "0", "type": "traffic_light"},
            {"id": "east", "x": "200", "y": "0", "type": "priority"},
            {"id": "north", "x": "0", "y": "200", "type": "priority"},
            {"id": "west", "x": "-200", "y": "0", "type": "priority"},
            {"id": "south", "x": "0", "y": "-200", "type": "priority"}
        ]
        node_file = self._create_node_file(nodes, scenario_dir, scenario_name)
        
        # Define edges
        edges = [
            {"id": "east_to_center", "from": "east", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_east", "from": "center", "to": "east", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "north_to_center", "from": "north", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_north", "from": "center", "to": "north", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "west_to_center", "from": "west", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_west", "from": "center", "to": "west", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "south_to_center", "from": "south", "to": "center", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"},
            {"id": "center_to_south", "from": "center", "to": "south", "numLanes": str(num_lanes), "speed": "13.89", "priority": "100"}
        ]
        edge_file = self._create_edge_file(edges, scenario_dir, scenario_name)
        
        # Create network file
        net_file = os.path.join(scenario_dir, f"{scenario_name}.net.xml")
        cmd = [
            "netconvert",
            "--node-files", node_file,
            "--edge-files", edge_file,
            "--output", net_file,
            "--no-turnarounds", "true",
            "--tls.green.time", "30",
            "--tls.yellow.time", "4",
            "--tls.red.time", "1"
        ]
        self._run_command(cmd)
        
        # Generate traffic for different levels in separate directories
        for level in self.traffic_levels.keys():
            level_dir = os.path.join(scenario_dir, level)
            os.makedirs(level_dir, exist_ok=True)
            
            route_file = os.path.join(scenario_dir, f"{scenario_name}_{level}.rou.xml")
            self._generate_traffic(net_file, route_file, level)
            
            # Create separate config file for each traffic level
            self._create_config_file(net_file, route_file, scenario_dir, scenario_name, level)
        
        return scenario_dir
    
    def create_spider_network(self, num_arms=4, num_lanes=1, scenario_name=None):
        """Create a spider network using netgenerate and then customize it with traffic lights at all intersections."""
        if scenario_name is None:
            scenario_name = f"spider_{num_arms}arms_{num_lanes}lane"
        
        scenario_dir = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Use netgenerate to create a base spider network
        base_net_file = os.path.join(scenario_dir, f"{scenario_name}_base.net.xml")
        cmd = [
            "netgenerate",
            "--spider",
            "--spider.arm-number", str(num_arms),
            "--spider.circle-number", "3",
            "--spider.space-radius", "100",
            "--default.lanenumber", str(num_lanes),
            "--output-file", base_net_file
        ]
        self._run_command(cmd)
        
        # Post-process with netconvert to set all junctions to traffic lights
        net_file = os.path.join(scenario_dir, f"{scenario_name}.net.xml")
        cmd = [
            "netconvert",
            "--sumo-net-file", base_net_file,
            "--output-file", net_file,
            "--tls.guess", "true",
            "--tls.guess.threshold", "1",  # Set a low threshold to ensure all intersections get traffic lights
            "--tls.green.time", "30",
            "--tls.yellow.time", "4",
            "--tls.red.time", "1"
        ]
        self._run_command(cmd)
        
        # Generate traffic for different levels in separate directories
        for level in self.traffic_levels.keys():
            level_dir = os.path.join(scenario_dir, level)
            os.makedirs(level_dir, exist_ok=True)
            
            route_file = os.path.join(scenario_dir, f"{scenario_name}_{level}.rou.xml")
            self._generate_traffic(net_file, route_file, level)
            
            # Create separate config file for each traffic level
            self._create_config_file(net_file, route_file, scenario_dir, scenario_name, level)
        
        return scenario_dir
    
    def create_grid_network(self, grid_size=3, num_lanes=1, scenario_name=None):
        """Create a grid network with specified size and lanes, with traffic lights at T-junctions."""
        if scenario_name is None:
            scenario_name = f"grid_{grid_size}x{grid_size}_{num_lanes}lane"
        
        scenario_dir = os.path.join(self.output_dir, scenario_name)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Parameters for grid network
        node_distance = 200  # Distance between nodes
        
        # Create nodes
        nodes = []
        for i in range(grid_size):
            for j in range(grid_size):
                x = i * node_distance
                y = j * node_distance
                node_id = f"node_{i}_{j}"
                
                # Check if node is at a corner (2-way intersection)
                is_corner = (i == 0 and j == 0) or (i == 0 and j == grid_size-1) or \
                            (i == grid_size-1 and j == 0) or (i == grid_size-1 and j == grid_size-1)
                
                # Corners remain priority, all others (T-junctions and 4-way) are traffic lights
                if is_corner:
                    node_type = "priority"
                else:
                    node_type = "traffic_light"
                
                nodes.append({"id": node_id, "x": str(x), "y": str(y), "type": node_type})
        
        node_file = self._create_node_file(nodes, scenario_dir, scenario_name)
        
        # Create edges
        edges = []
        for i in range(grid_size):
            for j in range(grid_size):
                # Horizontal edges
                if i < grid_size - 1:
                    # Forward edge (left to right)
                    edges.append({
                        "id": f"edge_{i}_{j}_to_{i+1}_{j}",
                        "from": f"node_{i}_{j}",
                        "to": f"node_{i+1}_{j}",
                        "numLanes": str(num_lanes),
                        "speed": "13.89"
                    })
                    # Backward edge (right to left)
                    edges.append({
                        "id": f"edge_{i+1}_{j}_to_{i}_{j}",
                        "from": f"node_{i+1}_{j}",
                        "to": f"node_{i}_{j}",
                        "numLanes": str(num_lanes),
                        "speed": "13.89"
                    })
                
                # Vertical edges
                if j < grid_size - 1:
                    # Forward edge (bottom to top)
                    edges.append({
                        "id": f"edge_{i}_{j}_to_{i}_{j+1}",
                        "from": f"node_{i}_{j}",
                        "to": f"node_{i}_{j+1}",
                        "numLanes": str(num_lanes),
                        "speed": "13.89"
                    })
                    # Backward edge (top to bottom)
                    edges.append({
                        "id": f"edge_{i}_{j+1}_to_{i}_{j}",
                        "from": f"node_{i}_{j+1}",
                        "to": f"node_{i}_{j}",
                        "numLanes": str(num_lanes),
                        "speed": "13.89"
                    })
        
        edge_file = self._create_edge_file(edges, scenario_dir, scenario_name)
        
        # Create network file
        net_file = os.path.join(scenario_dir, f"{scenario_name}.net.xml")
        cmd = [
            "netconvert",
            "--node-files", node_file,
            "--edge-files", edge_file,
            "--output", net_file,
            "--no-turnarounds", "true",
            "--tls.green.time", "30",
            "--tls.yellow.time", "4",
            "--tls.red.time", "1"
        ]
        self._run_command(cmd)
        
        # Generate traffic for different levels in separate directories
        for level in self.traffic_levels.keys():
            level_dir = os.path.join(scenario_dir, level)
            os.makedirs(level_dir, exist_ok=True)
            
            route_file = os.path.join(scenario_dir, f"{scenario_name}_{level}.rou.xml")
            self._generate_traffic(net_file, route_file, level)
            
            # Create separate config file for each traffic level
            self._create_config_file(net_file, route_file, scenario_dir, scenario_name, level)
        
        return scenario_dir

    def create_all_scenarios(self):
        """Create all scenarios for curriculum learning."""
        print("Creating curriculum scenarios for ATSC training...")
        
        print("\nGenerating T-junction scenarios...")
        for lanes in range(1, 3):
            print(f"  - T-junction with {lanes} lane(s)")
            self.create_t_junction(num_lanes=lanes)
        
        print("\nGenerating Four-way junction scenarios...")
        for lanes in range(1, 3):
            print(f"  - Four-way junction with {lanes} lane(s)")
            self.create_four_way_junction(num_lanes=lanes)
        
        print("\nGenerating Spider network scenarios...")
        for arms in range(5, 7):
            for lanes in range(1, 3):
                print(f"  - Spider network with {arms} arms and {lanes} lane(s)")
                self.create_spider_network(num_arms=arms, num_lanes=lanes)
        
        print("\nGenerating Grid network scenarios...")
        for size in range(4, 6):
            for lanes in range(1, 3):
                print(f"  - Grid network {size}x{size} with {lanes} lane(s)")
                self.create_grid_network(grid_size=size, num_lanes=lanes)
        
        print(f"\nAll scenarios created in '{self.output_dir}' directory.")


if __name__ == "__main__":
    # Example usage
    generator = CurriculumGenerator(output_dir="curriculum_scenarios")
    
    # To create all scenarios
    generator.create_all_scenarios()
    
    # To create individual scenarios:
    # generator.create_t_junction(num_lanes=2)
    # generator.create_four_way_junction(num_lanes=3)
    # generator.create_spider_network(num_arms=5, num_lanes=2)
    # generator.create_grid_network(grid_size=4, num_lanes=2)
