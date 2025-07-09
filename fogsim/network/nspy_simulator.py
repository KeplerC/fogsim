import simpy
import time
from typing import Dict, Any, Optional, Callable, List, Tuple, Set
import uuid
import logging

from ns.scheduler.virtual_clock import VirtualClockServer
from ns.packet.sink import PacketSink
from ns.packet.packet import Packet

# Set up logging
logger = logging.getLogger(__name__)

class PacketTracker:
    """Manages packet tracking and message delivery."""
    
    def __init__(self):
        self.pending_packets: Dict[str, Tuple[Any, float, int]] = {}  # msg_id -> (message, sent_time, flow_id)
        self.ready_messages: Dict[str, Any] = {}  # msg_id -> message
        self.last_checked_arrivals: Dict[int, int] = {}  # flow_id -> last checked arrival index
    
    def add_pending_packet(self, msg_id: str, message: Any, sent_time: float, flow_id: int) -> None:
        """Add a packet to pending tracking."""
        self.pending_packets[msg_id] = (message, sent_time, flow_id)
        if flow_id not in self.last_checked_arrivals:
            self.last_checked_arrivals[flow_id] = 0
    
    def mark_packet_delivered(self, msg_id: str) -> None:
        """Mark a packet as delivered and ready."""
        if msg_id in self.pending_packets:
            message, _, _ = self.pending_packets[msg_id]
            self.ready_messages[msg_id] = message
            del self.pending_packets[msg_id]
    
    def get_ready_messages(self) -> List[Any]:
        """Get and clear ready messages."""
        messages = list(self.ready_messages.values())
        self.ready_messages.clear()
        return messages
    
    def reset(self) -> None:
        """Reset all tracking."""
        self.pending_packets.clear()
        self.ready_messages.clear()
        self.last_checked_arrivals.clear()


class NSPyNetworkSimulator:
    """Network simulator implementation using ns.py with VirtualClock scheduler."""
    
    def __init__(self, source_rate: float = 4600.0, weights: List[int] = [1, 1], debug: bool = False,
                 env: Optional[simpy.Environment] = None, packet_tracker: Optional[PacketTracker] = None):
        """
        Initialize the network simulator with ns.py.
        
        Args:
            source_rate: Rate in bytes per second (default: 4600.0)
            weights: Weights for different flows (default: [1, 1])
            debug: Enable debug output (default: False)
            env: Optional SimPy environment for dependency injection
            packet_tracker: Optional packet tracker for dependency injection
        """
        self.env = env or simpy.Environment()
        self.source_rate = source_rate
        self.flow_weights = weights
        self.debug = debug
        self.vc_server = VirtualClockServer(self.env, source_rate, weights, debug=debug)
        self.sink = PacketSink(self.env)
        self.vc_server.out = self.sink
        
        # Message tracking
        self.packet_tracker = packet_tracker or PacketTracker()
        
        logger.info("NSPyNetworkSimulator initialized with rate=%f, weights=%s, debug=%s", 
                    source_rate, weights, debug)
        
    def register_packet(self, message: Any, flow_id: int = 0, size: float = 1000.0) -> str:
        """
        Register a packet to be sent through the network.
        
        Args:
            message: Message to send
            flow_id: Flow ID for the message (default: 0)
            size: Size of packet in bytes (default: 1000.0)
            
        Returns:
            str: Message ID
        """
        msg_id = str(uuid.uuid4())
        
        # Create and send packet through virtual clock server
        packet = Packet(self.env.now, size, flow_id=flow_id, packet_id=msg_id)
        
        # Store original message with the packet ID, sent time, and flow_id
        self.packet_tracker.add_pending_packet(msg_id, message, self.env.now, flow_id)
        
        # Send packet to server
        self.vc_server.put(packet)
        
        logger.info("Registered packet with ID=%s, flow_id=%d, size=%f, time=%f", 
                    msg_id, flow_id, size, self.env.now)
        
        return msg_id
    
    def run_until(self, time_point: float) -> None:
        """
        Run the simulation until specified time point.
        
        Args:
            time_point: Time to run until
        """
        # Skip running if time_point is not greater than current time
        if time_point <= self.env.now:
            logger.info("Skipping run_until as time_point=%f <= current_time=%f", 
                        time_point, self.env.now)
            return
            
        logger.info("Running simulation from %f to %f", self.env.now, time_point)
            
        # Run the simulation
        try:
            self.env.run(until=time_point)
        except simpy.core.EmptySchedule:
            logger.info("No more events to process in simulation")
            pass  # No more events to process
            
        # Check for new packet arrivals
        self._process_arrivals(time_point)
    
    def _process_arrivals(self, time_point: float) -> None:
        """Process packet arrivals up to the specified time point."""
        delivered_packet_ids = set()
        
        for flow_id in list(self.packet_tracker.last_checked_arrivals.keys()):
            # Check if this flow has any arrivals to process
            if flow_id in self.sink.arrivals:
                arrivals = self.sink.arrivals[flow_id]
                
                # Get last checked index for this flow
                last_idx = self.packet_tracker.last_checked_arrivals[flow_id]
                
                # Process any new arrivals since we last checked
                new_arrivals = 0
                for i in range(last_idx, len(arrivals)):
                    arrival_time = arrivals[i]
                    
                    # If this packet arrived by our time_point
                    if arrival_time <= time_point:
                        new_arrivals += 1
                        # In a real implementation, we would need a way to map arrival positions to
                        # specific packet IDs. Since we don't have access to the actual packet objects,
                        # we'll use a simplified approach: mark the first pending packet for this flow
                        # as delivered.
                        for packet_id, (message, sent_time, message_flow_id) in self.packet_tracker.pending_packets.items():
                            if message_flow_id == flow_id and packet_id not in delivered_packet_ids:
                                self.packet_tracker.mark_packet_delivered(packet_id)
                                delivered_packet_ids.add(packet_id)
                                logger.info("Packet ID=%s delivered at time=%f, latency=%f", 
                                            packet_id, arrival_time, arrival_time - sent_time)
                                break
                
                # Update the last checked index for next time
                self.packet_tracker.last_checked_arrivals[flow_id] = len(arrivals)
                logger.info("Flow %d: Processed %d new arrivals, total arrivals=%d", 
                             flow_id, new_arrivals, len(arrivals))
        
        logger.info("Delivered %d packets, %d still pending", 
                    len(delivered_packet_ids), len(self.packet_tracker.pending_packets))
    
    def get_ready_messages(self) -> List[Any]:
        """
        Get messages that are ready to be processed.
        
        Returns:
            List of ready messages
        """
        messages = self.packet_tracker.get_ready_messages()
        logger.info("Retrieved %d ready messages", len(messages))
        return messages
    
    def reset(self) -> None:
        """Reset the network simulator."""
        logger.info("Resetting network simulator")
        self.env = simpy.Environment()
        self.vc_server = VirtualClockServer(
            self.env, 
            self.source_rate, 
            self.flow_weights, 
            debug=self.debug
        )
        self.sink = PacketSink(self.env)
        self.vc_server.out = self.sink
        self.packet_tracker.reset()
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing network simulator")
        pass 