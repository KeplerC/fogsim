"""
Wallclock Network Simulator - Mode 2 (Real Clock + Simulated Network)

Uses the same ns.py network simulator but injects sleep() calls to simulate
network delays in wallclock time, ensuring synchronization with real time.
"""

import time
import logging
import simpy
from typing import Dict, Any, List, Optional
from .nspy_simulator import NSPyNetworkSimulator, PacketTracker

logger = logging.getLogger(__name__)


class WallclockNetworkSimulator(NSPyNetworkSimulator):
    """
    Wallclock-synchronized network simulator for Mode 2.
    
    Inherits from NSPyNetworkSimulator but adds sleep() calls to ensure
    network delays are experienced in real wallclock time.
    """
    
    def __init__(self, source_rate: float = 4600.0, weights: List[int] = [1, 1], 
                 debug: bool = False, link_delay: float = 0.0):
        """
        Initialize wallclock network simulator.
        
        Args:
            source_rate: Rate in bytes per second (default: 4600.0)
            weights: Weights for different flows (default: [1, 1])
            debug: Enable debug output (default: False)
            link_delay: Additional propagation delay in seconds (default: 0.0)
        """
        super().__init__(source_rate, weights, debug, None, None, link_delay)
        
        # Track wallclock time for synchronization
        self.last_process_time = time.time()
        self.start_time = time.time()
        
        logger.info("WallclockNetworkSimulator initialized for Mode 2 (real clock + simulated network)")
    
    def run_until(self, time_point: float) -> None:
        """
        Run simulation until specified time point with wallclock synchronization.
        
        This method overrides the parent to add sleep() for network delays.
        
        Args:
            time_point: Simulation time to run until
        """
        # Skip if time_point is not greater than current time
        if time_point <= self.env.now:
            logger.debug(f"Skipping run_until as time_point={time_point} <= current_time={self.env.now}")
            return
        
        # Run the ns.py simulation
        logger.debug(f"Running simulation from {self.env.now} to {time_point}")
        try:
            self.env.run(until=time_point)
        except simpy.core.EmptySchedule:
            logger.debug("No more events to process in simulation")
            pass
        
        # Process arrivals and calculate delays
        delivered_packets = self._process_arrivals_with_sleep(time_point)
        
        # Sleep for any remaining network processing time
        self._sync_with_wallclock(time_point)
    
    def _process_arrivals_with_sleep(self, time_point: float) -> List[str]:
        """
        Process packet arrivals and sleep for their simulated delays.
        
        Returns:
            List of delivered packet IDs
        """
        delivered_packet_ids = []
        current_wallclock = time.time()
        
        for flow_id in list(self.packet_tracker.last_checked_arrivals.keys()):
            if flow_id not in self.sink.arrivals:
                continue
                
            arrivals = self.sink.arrivals[flow_id]
            last_idx = self.packet_tracker.last_checked_arrivals[flow_id]
            
            # Process new arrivals
            for i in range(last_idx, len(arrivals)):
                arrival_time = arrivals[i]
                
                # Calculate total delivery time including link delay
                actual_delivery_time = arrival_time + self.link_delay
                
                if actual_delivery_time <= time_point:
                    # Find the corresponding packet
                    for packet_id, (message, sent_time, message_flow_id, packet_size) in list(self.packet_tracker.pending_packets.items()):
                        if message_flow_id == flow_id and packet_id not in delivered_packet_ids:
                            # Calculate the delay we need to simulate
                            simulated_delay = actual_delivery_time - sent_time
                            
                            # Sleep for the simulated delay if needed
                            if simulated_delay > 0:
                                elapsed_since_send = current_wallclock - self.start_time - sent_time
                                remaining_delay = simulated_delay - elapsed_since_send
                                
                                if remaining_delay > 0:
                                    logger.debug(f"Sleeping {remaining_delay*1000:.1f}ms for packet {packet_id}")
                                    time.sleep(remaining_delay)
                            
                            # Mark packet as delivered
                            if isinstance(message, dict):
                                message['latency'] = simulated_delay * 1000  # Convert to ms
                            self.packet_tracker.mark_packet_delivered(packet_id)
                            delivered_packet_ids.append(packet_id)
                            
                            logger.info(f"Packet {packet_id} delivered with {simulated_delay*1000:.1f}ms delay (wallclock synced)")
                            break
            
            # Update last checked index
            self.packet_tracker.last_checked_arrivals[flow_id] = len(arrivals)
        
        return delivered_packet_ids
    
    def _sync_with_wallclock(self, simulation_time: float) -> None:
        """
        Ensure wallclock time matches simulation time progression.
        
        Args:
            simulation_time: Current simulation time
        """
        # Calculate how much wallclock time should have passed
        target_wallclock = self.start_time + simulation_time
        current_wallclock = time.time()
        
        # Sleep if we're ahead of wallclock
        if current_wallclock < target_wallclock:
            sleep_duration = target_wallclock - current_wallclock
            logger.debug(f"Syncing with wallclock: sleeping {sleep_duration*1000:.1f}ms")
            time.sleep(sleep_duration)
    
    def reset(self) -> None:
        """Reset the simulator and wallclock tracking."""
        super().reset()
        self.start_time = time.time()
        self.last_process_time = time.time()
        logger.info("WallclockNetworkSimulator reset")