#!/usr/bin/env python3
"""
Example demonstrating the CloudSim visualization server with a Gym environment.

This example shows how to:
1. Set up and run the CloudSim visualization server
2. Create a Gym co-simulator
3. Wrap it with visualization capabilities
4. Run a simple simulation loop that sends frames to the visualization server

Requirements:
- gym or gymnasium (tested with gym==0.26.0 and gymnasium==0.29.1)
- Python 3.7+ (tested with Python 3.9)

Optional requirements for network simulation:
- ns3 (for real network simulation)
- networkx (for simple network simulation)

Usage:
    python visualization_example.py
"""

import os
import sys
import time
import logging
import numpy as np
import threading
import gymnasium as gym  # Use gymnasium (newer) or gym (older)
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CloudSim modules
from cloudsim.visualization.visualization_server import VisualizationServer
from cloudsim.visualization.simulator_wrapper import VisualizationCoSimulator
from cloudsim.environment.gym_co_simulator import GymCoSimulator

# Import a simple network simulator for demonstration
from simple_network_simulator import SimpleNetworkSimulator  # see below for implementation

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_example(args):
    """Run the visualization example."""
    # Create and start the visualization server
    logger.info("Starting visualization server on port %d", args.port)
    server = VisualizationServer(host=args.host, port=args.port)
    server_thread = server.run_in_thread()
    
    # Wait for server to start
    time.sleep(1)
    
    # Create a gym environment
    logger.info("Creating gym environment: %s", args.env)
    env = gym.make(args.env, render_mode="rgb_array")
    
    # Create a simple network simulator
    logger.info("Creating network simulator with latency %d ms", args.latency)
    network_sim = SimpleNetworkSimulator(latency=args.latency / 1000.0)  # convert ms to seconds
    
    # Create the GymCoSimulator
    logger.info("Creating GymCoSimulator with timestep %f", args.timestep)
    co_sim = GymCoSimulator(
        network_simulator=network_sim,
        gym_env=env,
        timestep=args.timestep
    )
    
    # Wrap with visualization capabilities
    logger.info("Creating visualization wrapper")
    viz_sim = VisualizationCoSimulator(
        co_simulator=co_sim,
        server_url=f"http://{args.host}:{args.port}",
        simulation_id=f"gym_{args.env}",
        auto_connect=True
    )
    
    # Reset the environment
    logger.info("Resetting environment")
    observation = viz_sim.reset()
    
    # Run the simulation loop
    total_reward = 0
    
    logger.info("Starting simulation loop")
    try:
        for step in range(args.steps):
            # Sample a random action
            action = env.action_space.sample()
            
            # Step the environment
            observation, reward, done, info = viz_sim.step(action)
            total_reward += reward
            
            logger.info(f"Step {step}/{args.steps}, reward: {reward:.4f}, total: {total_reward:.4f}")
            
            if done:
                logger.info(f"Episode finished after {step+1} steps with total reward {total_reward:.4f}")
                observation = viz_sim.reset()
                total_reward = 0
            
            # Slow down the simulation for better visualization
            time.sleep(args.delay)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping simulation")
    
    finally:
        # Close the environment
        logger.info("Closing environment")
        viz_sim.close()
        
        # Note: The server thread is daemonic and will close automatically when the script exits
        logger.info("Example completed")

# Simple network simulator for demonstration
class SimpleNetworkSimulator:
    """A simple network simulator that introduces delay but doesn't do full network simulation."""
    
    def __init__(self, latency=0.1, packet_loss=0.0, bandwidth=float('inf')):
        """
        Initialize the simple network simulator.
        
        Args:
            latency: One-way latency in seconds
            packet_loss: Packet loss probability (0.0 to 1.0)
            bandwidth: Bandwidth in bytes per second (default: unlimited)
        """
        self.latency = latency
        self.packet_loss = packet_loss
        self.bandwidth = bandwidth
        self.current_time = 0.0
        self.packets = {}  # packet_id -> (arrival_time, data)
        self.packet_counter = 0
        
        logger.info(f"SimpleNetworkSimulator initialized with latency={latency}s, " 
                   f"packet_loss={packet_loss}, bandwidth={bandwidth}B/s")
    
    def register_packet(self, data, flow_id=0, size=1000.0):
        """
        Register a packet for sending.
        
        Args:
            data: Packet data
            flow_id: Flow ID (not used in this simple simulator)
            size: Size of packet in bytes
            
        Returns:
            str: Packet ID
        """
        # Generate a unique packet ID
        self.packet_counter += 1
        packet_id = str(self.packet_counter)
        
        # Calculate when the packet will arrive
        transit_time = self.latency
        
        # If bandwidth is limited, add transmission delay
        if self.bandwidth < float('inf') and size > 0:
            transit_time += size / self.bandwidth
        
        # Simulate packet loss
        if np.random.random() < self.packet_loss:
            logger.info(f"Packet {packet_id} dropped (packet loss)")
            return packet_id
        
        # Schedule the packet
        arrival_time = self.current_time + transit_time
        self.packets[packet_id] = (arrival_time, data)
        
        logger.debug(f"Packet {packet_id} scheduled to arrive at {arrival_time}")
        
        return packet_id
    
    def run_until(self, time):
        """
        Run the network simulator until the specified time.
        
        Args:
            time: Time to run until
        """
        self.current_time = time
    
    def get_ready_messages(self):
        """
        Get messages that have arrived by the current time.
        
        Returns:
            list: List of message data
        """
        ready_messages = []
        packets_to_remove = []
        
        for packet_id, (arrival_time, data) in self.packets.items():
            if arrival_time <= self.current_time:
                ready_messages.append(data)
                packets_to_remove.append(packet_id)
        
        # Remove delivered packets
        for packet_id in packets_to_remove:
            del self.packets[packet_id]
        
        return ready_messages
    
    def reset(self):
        """Reset the network simulator."""
        self.current_time = 0.0
        self.packets.clear()
        self.packet_counter = 0
    
    def close(self):
        """Clean up resources."""
        pass
    
    # Network parameter adjustment methods
    def set_latency(self, latency):
        """Set the network latency (seconds)."""
        self.latency = latency
        logger.info(f"Network latency set to {latency}s")
        return True
    
    def set_packet_loss(self, packet_loss):
        """Set the packet loss probability (0.0 to 1.0)."""
        self.packet_loss = max(0.0, min(1.0, packet_loss))
        logger.info(f"Packet loss set to {self.packet_loss}")
        return True
    
    def set_bandwidth(self, bandwidth):
        """Set the bandwidth in bytes per second."""
        self.bandwidth = float(bandwidth) if bandwidth > 0 else float('inf')
        logger.info(f"Bandwidth set to {self.bandwidth}B/s")
        return True

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CloudSim visualization example')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host address to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on')
    parser.add_argument('--env', type=str, default='CartPole-v1',
                        help='Gym environment to use')
    parser.add_argument('--timestep', type=float, default=0.1,
                        help='Simulation timestep in seconds')
    parser.add_argument('--latency', type=int, default=50,
                        help='Network latency in milliseconds')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of simulation steps to run')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps for visualization')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_example(args) 