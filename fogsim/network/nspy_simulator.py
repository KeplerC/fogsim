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
        self.pending_packets: Dict[str, Tuple[Any, float, int, float]] = {}  # msg_id -> (message, sent_time, flow_id, packet_size)
        self.ready_messages: Dict[str, Any] = {}  # msg_id -> message
        self.last_checked_arrivals: Dict[int, int] = {}  # flow_id -> last checked arrival index
        # New: Track packet order and expected delivery times
        self.packet_send_order: Dict[int, List[Tuple[str, float, float]]] = {}  # flow_id -> [(packet_id, sent_time, expected_delivery)]
        self.delivered_packet_ids: Set[str] = set()  # Track which packets have been delivered
    
    def add_pending_packet(self, msg_id: str, message: Any, sent_time: float, flow_id: int, packet_size: float = 1000.0) -> None:
        """Add a packet to pending tracking."""
        self.pending_packets[msg_id] = (message, sent_time, flow_id, packet_size)
        if flow_id not in self.last_checked_arrivals:
            self.last_checked_arrivals[flow_id] = 0
        if flow_id not in self.packet_send_order:
            self.packet_send_order[flow_id] = []
        
        # Track send order for proper delivery mapping
        # Expected delivery is sent_time + estimated delay (will be updated when actual delivery happens)
        self.packet_send_order[flow_id].append((msg_id, sent_time, sent_time))
    
    def mark_packet_delivered(self, msg_id: str, actual_delivery_time: float = None) -> None:
        """Mark a packet as delivered and ready."""
        if msg_id in self.pending_packets and msg_id not in self.delivered_packet_ids:
            message, sent_time, flow_id, packet_size = self.pending_packets[msg_id]
            
            # Add delivery time info to message if it's a dict
            if isinstance(message, dict) and actual_delivery_time is not None:
                message['delivery_time'] = actual_delivery_time
                message['latency'] = (actual_delivery_time - sent_time) * 1000  # Convert to ms
            
            self.ready_messages[msg_id] = message
            self.delivered_packet_ids.add(msg_id)
            del self.pending_packets[msg_id]
            
            logger.debug(f"Packet {msg_id} delivered at {actual_delivery_time}, latency={((actual_delivery_time - sent_time) * 1000) if actual_delivery_time else 'unknown'}ms")
    
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
        self.packet_send_order.clear()
        self.delivered_packet_ids.clear()
        
    def get_oldest_pending_packet_for_flow(self, flow_id: int) -> Optional[str]:
        """Get the oldest pending packet ID for a specific flow."""
        if flow_id not in self.packet_send_order:
            return None
        
        # Find the oldest packet that hasn't been delivered yet
        for packet_id, sent_time, _ in self.packet_send_order[flow_id]:
            if packet_id in self.pending_packets and packet_id not in self.delivered_packet_ids:
                return packet_id
        return None


class NSPyNetworkSimulator:
    """Network simulator implementation using ns.py with VirtualClock scheduler."""
    
    def __init__(self, source_rate: float = 4600.0, weights: List[int] = [1, 1], debug: bool = False,
                 env: Optional[simpy.Environment] = None, packet_tracker: Optional[PacketTracker] = None,
                 link_delay: float = 0.0):
        """
        Initialize the network simulator with ns.py.
        
        Args:
            source_rate: Rate in bytes per second (default: 4600.0)
            weights: Weights for different flows (default: [1, 1])
            debug: Enable debug output (default: False)
            env: Optional SimPy environment for dependency injection
            packet_tracker: Optional packet tracker for dependency injection
            link_delay: Additional propagation delay in seconds (default: 0.0)
        """
        self.env = env or simpy.Environment()
        self.source_rate = source_rate
        self.flow_weights = weights
        self.debug = debug
        self.link_delay = link_delay
        self.vc_server = VirtualClockServer(self.env, source_rate, weights, debug=debug)
        self.sink = PacketSink(self.env)
        self.vc_server.out = self.sink
        
        # CRITICAL FIX: Start the VirtualClock server process
        self.server_process = self.env.process(self.vc_server.run())
        
        # Message tracking
        self.packet_tracker = packet_tracker or PacketTracker()
        
        logger.info("NSPyNetworkSimulator initialized with rate=%f, weights=%s, debug=%s, link_delay=%f", 
                    source_rate, weights, debug, link_delay)
        
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
        
        # Store original message with the packet ID, sent time, flow_id, and size
        self.packet_tracker.add_pending_packet(msg_id, message, self.env.now, flow_id, size)
        
        # Send packet to server
        self.vc_server.put(packet)
        
        # Enhanced logging
        msg_type = message.get('type', 'unknown') if isinstance(message, dict) else 'unknown'
        logger.info("Registered packet ID=%s, flow_id=%d, type=%s, size=%.0f, time=%.3f", 
                    msg_id, flow_id, msg_type, size, self.env.now)
        
        return msg_id
    
    def run_until(self, time_point: float) -> None:
        """
        Run the simulation until specified time point.
        
        Args:
            time_point: Time to run until
        """
        # Skip running if time_point is not greater than current time
        if time_point <= self.env.now:
            logger.debug("Skipping run_until as time_point=%.3f <= current_time=%.3f", 
                        time_point, self.env.now)
            # Still process arrivals even if we don't advance time
            self._process_arrivals(time_point)
            return
            
        pending_before = len(self.packet_tracker.pending_packets)
        logger.debug("Running simulation from %.3f to %.3f (pending packets: %d)", 
                    self.env.now, time_point, pending_before)
            
        # Run the simulation
        try:
            self.env.run(until=time_point)
        except simpy.core.EmptySchedule:
            logger.debug("No more events to process in simulation")
            pass  # No more events to process
            
        # Check for new packet arrivals
        self._process_arrivals(time_point)
        
        # Debug: Log packet status after processing
        pending_after = len(self.packet_tracker.pending_packets)
        ready_messages = len(self.packet_tracker.ready_messages)
        if pending_before != pending_after or ready_messages > 0:
            logger.info("Packet status: pending %d→%d, ready messages: %d", 
                       pending_before, pending_after, ready_messages)
    
    def _process_arrivals(self, time_point: float) -> None:
        """Process packet arrivals up to the specified time point."""
        delivered_count = 0
        
        # Check all flows that have sent packets
        all_flow_ids = set(self.packet_tracker.last_checked_arrivals.keys())
        # Also check flows that have arrivals in the sink
        if hasattr(self.sink, 'arrivals'):
            all_flow_ids.update(self.sink.arrivals.keys())
        
        for flow_id in all_flow_ids:
            # Check if this flow has any arrivals to process
            if flow_id in self.sink.arrivals and self.sink.arrivals[flow_id]:
                arrivals = self.sink.arrivals[flow_id]
                
                # Get last checked index for this flow
                last_idx = self.packet_tracker.last_checked_arrivals[flow_id]
                
                # Process any new arrivals since we last checked
                new_arrivals = 0
                for i in range(last_idx, len(arrivals)):
                    arrival_time = arrivals[i]
                    
                    # Add link delay to the arrival time
                    actual_delivery_time = arrival_time + self.link_delay
                    
                    # If this packet has been delivered (including link delay) by our time_point
                    if actual_delivery_time <= time_point:
                        new_arrivals += 1
                        
                        # FIXED: Get the oldest pending packet for this flow (FIFO order)
                        oldest_packet_id = self.packet_tracker.get_oldest_pending_packet_for_flow(flow_id)
                        
                        if oldest_packet_id is not None:
                            # Mark this specific packet as delivered
                            self.packet_tracker.mark_packet_delivered(oldest_packet_id, actual_delivery_time)
                            delivered_count += 1
                            
                            # Get packet info for logging
                            if oldest_packet_id in self.packet_tracker.pending_packets:
                                message, sent_time, message_flow_id, packet_size = self.packet_tracker.pending_packets[oldest_packet_id]
                            else:
                                # Packet was just delivered, reconstruct info for logging
                                sent_time = arrival_time - self.link_delay  # Approximate
                                message_flow_id = flow_id
                                packet_size = 1000.0
                            
                            logger.info("Packet ID=%s delivered at time=%.3f, total_latency=%.1fms (queue=%.1fms + link=%.1fms)", 
                                        oldest_packet_id, actual_delivery_time, 
                                        (actual_delivery_time - sent_time) * 1000,
                                        (arrival_time - sent_time) * 1000, 
                                        self.link_delay * 1000)
                        else:
                            logger.warning("Arrival detected but no pending packet found for flow %d at time %.3f", 
                                          flow_id, actual_delivery_time)
                
                # Update the last checked index for next time
                self.packet_tracker.last_checked_arrivals[flow_id] = len(arrivals)
                logger.info("Flow %d: Processed %d new arrivals, %d delivered, total arrivals=%d", 
                             flow_id, new_arrivals, new_arrivals, len(arrivals))
        
        logger.info("Delivered %d packets, %d still pending", 
                    delivered_count, len(self.packet_tracker.pending_packets))
    
    def get_ready_messages(self) -> List[Any]:
        """
        Get messages that are ready to be processed.
        
        Returns:
            List of ready messages
        """
        messages = self.packet_tracker.get_ready_messages()
        if messages:
            logger.info("Retrieved %d ready messages", len(messages))
            for i, msg in enumerate(messages):
                if isinstance(msg, dict) and 'latency' in msg:
                    logger.debug("Message %d: type=%s, latency=%.1fms", 
                               i, msg.get('type', 'unknown'), msg['latency'])
        return messages
    
    def reset(self) -> None:
        """Reset the network simulator."""
        logger.info("Resetting network simulator")
        # Don't create a new environment - keep the existing one but reset its time
        self.env = simpy.Environment()
        self.vc_server = VirtualClockServer(
            self.env, 
            self.source_rate, 
            self.flow_weights, 
            debug=self.debug
        )
        self.sink = PacketSink(self.env)
        self.vc_server.out = self.sink
        
        # CRITICAL FIX: Restart the VirtualClock server process
        self.server_process = self.env.process(self.vc_server.run())
        
        self.packet_tracker.reset()
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing network simulator")
        pass 