from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List, Protocol
import numpy as np
import time
import logging

from .time_backend import UnifiedTimeManager, SimulationMode, TimeSubscriber
from .message_passing import MessageBus, MessageHandler as MessageHandlerInterface, TimedMessage
from .network_control import NetworkControlManager, NetworkConfig

# Set up logging
logger = logging.getLogger(__name__)


class NetworkSimulatorProtocol(Protocol):
    """Protocol defining the interface for network simulators."""
    
    def run_until(self, time: float) -> None:
        """Run simulation until specified time."""
        ...
    
    def get_ready_messages(self) -> List[Any]:
        """Get messages ready for processing."""
        ...
    
    def register_packet(self, message: Any, flow_id: int, size: float) -> str:
        """Register a packet for transmission."""
        ...
    
    def close(self) -> None:
        """Clean up resources."""
        ...


class LegacyTimeManager:
    """Legacy time manager for backward compatibility."""
    
    def __init__(self, unified_manager: UnifiedTimeManager):
        self.unified_manager = unified_manager
        self.timestep = unified_manager.timestep
    
    def advance_time(self) -> None:
        """Advance the simulation time by one timestep."""
        self.unified_manager.advance_time()
        logger.info("Advanced simulation time to %f", self.current_time)
    
    def get_current_time(self) -> float:
        """Get the current simulation time."""
        return self.unified_manager.now()
    
    def get_timestep(self) -> float:
        """Get the simulation timestep."""
        return self.timestep
    
    def reset_time(self) -> None:
        """Reset time to initial state."""
        self.unified_manager.reset()
    
    @property
    def current_time(self) -> float:
        return self.unified_manager.now()
    
    @current_time.setter
    def current_time(self, value: float) -> None:
        # This is a compatibility hack - should not be used
        logger.warning("Direct time setting is deprecated")


class LegacyMessageHandler:
    """Legacy message handler for backward compatibility."""
    
    CLIENT_TO_SERVER_FLOW = 0
    SERVER_TO_CLIENT_FLOW = 1
    
    def __init__(self, message_bus: MessageBus, node_id: str):
        self.message_bus = message_bus
        self.node_id = node_id
        self.pending_messages = []
        
        # Register a handler to collect messages
        handler = LegacyMessageCollector(self.pending_messages)
        self.message_bus.register_handler(node_id, handler)
    
    def process_messages(self, current_time: float) -> List[Any]:
        """Process any messages that should be visible at the current timestep."""
        logger.info("Processing network messages at time %f", current_time)
        
        # Messages are automatically processed by the message bus
        # Return and clear pending messages
        messages = list(self.pending_messages)
        self.pending_messages.clear()
        
        logger.info("Retrieved %d messages ready for processing", len(messages))
        return messages
    
    def send_message(self, message: Any, flow_id: int = 0, size: float = 1000.0) -> str:
        """Send a message through the network simulator."""
        logger.info("Sending message with flow_id=%d, size=%f", flow_id, size)
        
        # Map flow_id to sender/receiver
        if flow_id == self.CLIENT_TO_SERVER_FLOW:
            sender = f"{self.node_id}_client"
            receiver = f"{self.node_id}_server"
        else:
            sender = f"{self.node_id}_server"
            receiver = f"{self.node_id}_client"
        
        # Calculate delay based on size (simple model)
        delay = size / 1e6  # 1MB/s for now
        
        self.message_bus.send(sender, receiver, message, delay)
        return f"{sender}->{receiver}-{current_time}"


class LegacyMessageCollector(MessageHandlerInterface):
    """Collects messages for legacy interface."""
    
    def __init__(self, message_list: List[Any]):
        self.message_list = message_list
    
    def handle_message(self, message: TimedMessage) -> None:
        """Collect message payload."""
        self.message_list.append(message.payload)


class BaseCoSimulator(TimeSubscriber, ABC):
    """Base class for co-simulation between robotics and network simulation."""
    
    def __init__(self, network_simulator: Any, robotics_simulator: Any, 
                 timestep: float = 0.1, mode: SimulationMode = SimulationMode.VIRTUAL):
        """
        Initialize the co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            robotics_simulator: Instance of robotics simulator (e.g., gym, carla)
            timestep: Unified simulation timestep in seconds (default: 0.1)
            mode: Simulation mode (VIRTUAL, SIMULATED_NET, or REAL_NET)
        """
        self.network_simulator = network_simulator
        self.robotics_simulator = robotics_simulator
        self.mode = mode
        
        # Set a single unified timestep for all simulation components
        self.timestep = timestep
        
        # Initialize unified time manager
        self._unified_time_manager = UnifiedTimeManager(mode, timestep)
        self._unified_time_manager.register_subscriber(self)
        
        # Initialize message bus
        self._message_bus = MessageBus(self._unified_time_manager, network_simulator)
        
        # Initialize network control
        self._network_control = NetworkControlManager(mode, network_simulator=network_simulator)
        
        # Legacy compatibility
        self._time_manager = LegacyTimeManager(self._unified_time_manager)
        self._message_handler = LegacyMessageHandler(self._message_bus, "cosim")
        
        # Define flow IDs for different message types (for backward compatibility)
        self.CLIENT_TO_SERVER_FLOW = LegacyMessageHandler.CLIENT_TO_SERVER_FLOW
        self.SERVER_TO_CLIENT_FLOW = LegacyMessageHandler.SERVER_TO_CLIENT_FLOW
        
        logger.info("BaseCoSimulator initialized with mode: %s, timestep: %f", mode.value, self.timestep)
        
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator
            
        Returns:
            Tuple containing:
            - observation: Current state observation
            - reward: Reward for the current step
            - done: Whether the episode is done
            - info: Additional information
        """
        pass
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset both simulators to initial state.
        
        Returns:
            Initial observation
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human') -> None:
        """
        Render the current state of the simulation.
        
        Args:
            mode: Rendering mode
        """
        pass
    
    def _process_network_messages(self) -> List[Any]:
        """
        Process any messages that should be visible at the current timestep.
        
        Returns:
            List of messages ready for processing
        """
        return self._message_handler.process_messages(self.current_time)
    
    @abstractmethod
    def _handle_message(self, message: Any) -> None:
        """
        Handle a received message from the network simulator.
        
        Args:
            message: The received message
        """
        pass
    
    def _send_message(self, message: Any, flow_id: int = 0, size: float = 1000.0) -> str:
        """
        Send a message through the network simulator.
        
        Args:
            message: Message to send
            flow_id: Flow ID (default: 0, client to server)
            size: Size of message in bytes (default: 1000.0)
            
        Returns:
            str: Message ID
        """
        return self._message_handler.send_message(message, flow_id, size)
    
    def _advance_time(self) -> None:
        """Advance the simulation time by one timestep."""
        self._unified_time_manager.advance_time()
        
    def get_current_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns:
            float: Current simulation time in seconds
        """
        return self._unified_time_manager.now()
    
    def get_timestep(self) -> float:
        """
        Get the simulation timestep.
        
        Returns:
            float: Unified simulation timestep in seconds
        """
        return self.timestep
    
    @property
    def current_time(self) -> float:
        """Property for backward compatibility."""
        return self._unified_time_manager.now()
    
    @current_time.setter
    def current_time(self, value: float) -> None:
        """Setter for backward compatibility."""
        logger.warning("Direct time setting is deprecated in new architecture")
    
    def sync_to_time(self, time: float) -> None:
        """TimeSubscriber interface - called when time is synchronized."""
        # Process any pending network messages
        self._process_network_messages()
    
    def configure_network(self, config: NetworkConfig) -> None:
        """Configure network parameters (for modes 2 and 3)."""
        self._network_control.configure(config)
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing simulators")
        self._unified_time_manager.unregister_subscriber(self)
        self._network_control.reset()
        self.robotics_simulator.close()
        self.network_simulator.close() 