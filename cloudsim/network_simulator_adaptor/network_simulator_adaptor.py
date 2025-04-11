#!/usr/bin/env python3

import logging
import os
import sys

# Add the parent directory to the path for importing protos
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('network_simulator_adaptor')

class NetworkSimulatorAdaptor:
    def __init__(self):
        # Get configuration from environment variables
        self.adaptor_id = os.environ.get('ADAPTOR_ID', 'network_simulator_adaptor')
        self.network_simulator_type = os.environ.get('NETWORK_SIMULATOR_TYPE', 'simple')  # 'simple', 'ns3', etc.
        
        # Initialize network simulator based on type
        self.init_network_simulator()
        
        logger.info(f"Network Simulator Adaptor {self.adaptor_id} initialized with {self.network_simulator_type} simulator")
    
    def init_network_simulator(self):
        if self.network_simulator_type == 'simple':
            # Simple constant latency simulator
            self.network_simulator = SimpleNetworkSimulator()
        elif self.network_simulator_type == 'ns3':
            # NS3 simulator (placeholder)
            self.network_simulator = NS3NetworkSimulator()
        else:
            logger.error(f"Unknown network simulator type: {self.network_simulator_type}")
            raise ValueError(f"Unknown network simulator type: {self.network_simulator_type}")
    
    def calculate_delay(self, message):
        """
        Calculate network delay for a message in nanoseconds
        
        Args:
            message (dict): The message to calculate delay for
            
        Returns:
            int: Delay in nanoseconds
        """
        # Extract relevant information from message
        source = message.get('source_id', '')
        destination = message.get('destination_id', '')
        
        # Calculate message size
        if 'binary_data' in message:
            message_size = len(message['binary_data'])
        else:
            # Estimate size if binary data not available
            message_size = 1024  # Default 1KB
        
        # Calculate latency using network simulator
        latency_ns = self.network_simulator.calculate_latency(source, destination, message_size)
        
        logger.info(f"Calculated network delay: {latency_ns}ns from {source} to {destination}")
        return latency_ns


# Simple network simulator with constant latency
class SimpleNetworkSimulator:
    def __init__(self):
        self.base_latency_ns = 1000000  # 1ms base latency
        self.bandwidth_bps = 1000000000  # 1Gbps
    
    def calculate_latency(self, source, destination, packet_size_bytes):
        # Simple latency model: base_latency + (packet_size / bandwidth)
        transmission_latency_ns = (packet_size_bytes * 8 * 1000000000) // self.bandwidth_bps
        return self.base_latency_ns + transmission_latency_ns


# NS3 network simulator (placeholder)
class NS3NetworkSimulator:
    def __init__(self):
        # This would initialize the NS3 simulator, potentially via a subprocess or API
        logger.info("Initializing NS3 Network Simulator")
        self.ns3_process = None
    
    def calculate_latency(self, source, destination, packet_size_bytes):
        # This would call into the NS3 simulator to calculate latency
        # Placeholder implementation
        logger.info(f"NS3 calculating latency from {source} to {destination} for {packet_size_bytes} bytes")
        # Call into NS3 API or use IPC to get the latency
        return 5000000  # 5ms in nanoseconds 