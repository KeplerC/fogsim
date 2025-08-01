"""
Time-Aware Message Passing System for FogSim

This module provides message passing capabilities that respect the different
timing modes and ensure proper synchronization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from queue import PriorityQueue, Queue
import threading
import logging
from abc import ABC, abstractmethod
import time

from .time_backend import SimulationMode, UnifiedTimeManager


logger = logging.getLogger(__name__)


@dataclass
class TimedMessage:
    """Message with timing information"""
    payload: Any
    send_time: float
    receive_time: float
    sender_id: str
    receiver_id: str
    message_type: str = "data"
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # For real network mode
    wallclock_send: Optional[float] = None
    wallclock_receive: Optional[float] = None
    
    def __lt__(self, other):
        # For priority queue - earlier receive time or higher priority first
        if self.receive_time != other.receive_time:
            return self.receive_time < other.receive_time
        return self.priority > other.priority


class MessageHandler(ABC):
    """Abstract interface for message handlers"""
    
    @abstractmethod
    def handle_message(self, message: TimedMessage) -> None:
        """Process a received message"""
        pass


class MessageScheduler:
    """
    Schedules message delivery based on simulation mode
    """
    
    def __init__(self, time_manager: UnifiedTimeManager, mode: SimulationMode):
        self.time_manager = time_manager
        self.mode = mode
        self.message_queue = PriorityQueue()
        self.handlers: Dict[str, MessageHandler] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # For real network mode
        self.real_network_queue: Optional[Queue] = None
        if mode == SimulationMode.REAL_NET:
            self.real_network_queue = Queue()
            self._start_real_network_thread()
    
    def register_handler(self, receiver_id: str, handler: MessageHandler) -> None:
        """Register a message handler for a receiver"""
        with self._lock:
            self.handlers[receiver_id] = handler
    
    def send_message(self, message: TimedMessage) -> None:
        """Schedule a message for delivery"""
        current_time = self.time_manager.now()
        
        if self.mode == SimulationMode.VIRTUAL:
            # Pure virtual time scheduling
            self._schedule_virtual_delivery(message)
            
        elif self.mode == SimulationMode.SIMULATED_NET:
            # Wallclock with simulated delays
            message.wallclock_send = time.time()
            self._schedule_simulated_delivery(message)
            
        else:  # REAL_NET
            # Real network delivery
            message.wallclock_send = time.time()
            self._schedule_real_delivery(message)
    
    def process_messages(self) -> None:
        """Process all messages ready for delivery"""
        current_time = self.time_manager.now()
        
        with self._lock:
            # Process messages that should be delivered by now
            delivered = []
            
            while not self.message_queue.empty():
                message = self.message_queue.get()
                
                if message.receive_time <= current_time:
                    # Deliver the message
                    self._deliver_message(message)
                    delivered.append(message)
                else:
                    # Put it back, not time yet
                    self.message_queue.put(message)
                    break
            
            if delivered:
                logger.debug(f"Delivered {len(delivered)} messages at time {current_time}")
    
    def _schedule_virtual_delivery(self, message: TimedMessage) -> None:
        """Schedule delivery in virtual time"""
        # Schedule the delivery event
        self.time_manager.schedule_event(
            message.receive_time,
            lambda: self._deliver_message(message),
            priority=message.priority
        )
        
        # Also add to queue for process_messages
        with self._lock:
            self.message_queue.put(message)
    
    def _schedule_simulated_delivery(self, message: TimedMessage) -> None:
        """Schedule delivery with simulated network delay"""
        # Calculate when the message should be delivered in simulation time
        delivery_time = message.receive_time
        
        # Add to priority queue
        with self._lock:
            self.message_queue.put(message)
        
        # For simulated network mode, we rely on process_messages
        # being called regularly as time advances
    
    def _schedule_real_delivery(self, message: TimedMessage) -> None:
        """Schedule delivery through real network"""
        # In real network mode, send immediately
        # The actual network delay happens naturally
        if self.real_network_queue:
            self.real_network_queue.put(message)
    
    def _deliver_message(self, message: TimedMessage) -> None:
        """Deliver message to handler"""
        handler = self.handlers.get(message.receiver_id)
        if handler:
            try:
                handler.handle_message(message)
            except Exception as e:
                logger.error(f"Error handling message: {e}")
        else:
            logger.warning(f"No handler for receiver {message.receiver_id}")
    
    def _start_real_network_thread(self) -> None:
        """Start thread for real network message handling"""
        def network_thread():
            while not self._stop_event.is_set():
                try:
                    # Get message from queue with timeout
                    message = self.real_network_queue.get(timeout=0.1)
                    
                    # In real network mode, deliver immediately
                    # The network delay already happened
                    message.wallclock_receive = time.time()
                    self._deliver_message(message)
                    
                except:
                    continue
        
        thread = threading.Thread(target=network_thread, daemon=True)
        thread.start()
    
    def stop(self) -> None:
        """Stop the message scheduler"""
        self._stop_event.set()


class MessageBus:
    """
    High-level message bus that integrates with time management
    """
    
    def __init__(self, time_manager: UnifiedTimeManager, 
                 network_simulator=None):
        self.time_manager = time_manager
        self.mode = time_manager.mode
        self.scheduler = MessageScheduler(time_manager, self.mode)
        self.network_simulator = network_simulator
        
        # Subscribe to time updates
        self.time_manager.register_subscriber(self)
    
    def sync_to_time(self, time: float) -> None:
        """TimeSubscriber interface - process messages up to current time"""
        self.scheduler.process_messages()
    
    def send(self, sender_id: str, receiver_id: str, payload: Any,
             delay: float = 0.0, priority: int = 0,
             message_type: str = "data") -> None:
        """
        Send a message with optional delay
        
        Args:
            sender_id: ID of the sender
            receiver_id: ID of the receiver
            payload: Message payload
            delay: Network delay in seconds
            priority: Message priority (higher = more important)
            message_type: Type of message
        """
        current_time = self.time_manager.now()
        
        # Create timed message
        message = TimedMessage(
            payload=payload,
            send_time=current_time,
            receive_time=current_time + delay,
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            priority=priority
        )
        
        # If we have a network simulator, use it to calculate delay
        if self.network_simulator and self.mode == SimulationMode.SIMULATED_NET:
            # Let network simulator determine actual delay
            sim_delay = self._calculate_network_delay(message)
            message.receive_time = current_time + sim_delay
        
        # Send through scheduler
        self.scheduler.send_message(message)
    
    def register_handler(self, receiver_id: str, handler: MessageHandler) -> None:
        """Register a message handler"""
        self.scheduler.register_handler(receiver_id, handler)
    
    def _calculate_network_delay(self, message: TimedMessage) -> float:
        """Calculate network delay using network simulator"""
        if hasattr(self.network_simulator, 'calculate_delay'):
            # Get delay from network simulator
            base_delay = self.network_simulator.calculate_delay(
                message.sender_id, 
                message.receiver_id,
                len(str(message.payload))
            )
            return base_delay
        return 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get message bus statistics"""
        with self.scheduler._lock:
            return {
                "mode": self.mode.value,
                "current_time": self.time_manager.now(),
                "queued_messages": self.scheduler.message_queue.qsize(),
                "registered_handlers": len(self.scheduler.handlers)
            }


class SimpleMessageHandler(MessageHandler):
    """Simple message handler that stores received messages"""
    
    def __init__(self):
        self.received_messages: List[TimedMessage] = []
        self.message_callback: Optional[Callable] = None
    
    def handle_message(self, message: TimedMessage) -> None:
        """Store received message"""
        self.received_messages.append(message)
        
        if self.message_callback:
            self.message_callback(message)
    
    def set_callback(self, callback: Callable) -> None:
        """Set callback for message reception"""
        self.message_callback = callback
    
    def clear(self) -> None:
        """Clear received messages"""
        self.received_messages.clear()