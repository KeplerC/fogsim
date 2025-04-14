"""
Simple network simulator for CloudSim examples.

This module provides a basic network simulator that introduces delay and
packet loss but doesn't do full network simulation. It's useful for testing
and demonstration purposes when a full network simulator like ns-3 is not available.
"""

import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

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