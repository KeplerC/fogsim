from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List, Protocol
import numpy as np
import time
import logging

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


class TimeManager:
    """Manages simulation time progression."""
    
    def __init__(self, timestep: float):
        self.timestep = timestep
        self.current_time = timestep
        self._last_step_time: Optional[float] = None
    
    def advance_time(self) -> None:
        """Advance the simulation time by one timestep."""
        self.current_time += self.timestep
        logger.info("Advanced simulation time to %f", self.current_time)
    
    def get_current_time(self) -> float:
        """Get the current simulation time."""
        return self.current_time
    
    def get_timestep(self) -> float:
        """Get the simulation timestep."""
        return self.timestep
    
    def reset_time(self) -> None:
        """Reset time to initial state."""
        self.current_time = self.timestep
        self._last_step_time = None


class MessageHandler:
    """Handles message sending and receiving through network simulator."""
    
    CLIENT_TO_SERVER_FLOW = 0
    SERVER_TO_CLIENT_FLOW = 1
    
    def __init__(self, network_simulator: NetworkSimulatorProtocol):
        self.network_simulator = network_simulator
    
    def process_messages(self, current_time: float) -> List[Any]:
        """Process any messages that should be visible at the current timestep."""
        logger.info("Processing network messages at time %f", current_time)
        self.network_simulator.run_until(current_time)
        messages = self.network_simulator.get_ready_messages()
        logger.info("Retrieved %d messages ready for processing", len(messages))
        return messages
    
    def send_message(self, message: Any, flow_id: int = 0, size: float = 1000.0) -> str:
        """Send a message through the network simulator."""
        logger.info("Sending message with flow_id=%d, size=%f", flow_id, size)
        msg_id = self.network_simulator.register_packet(message, flow_id, size)
        logger.info("Message sent with ID: %s", msg_id)
        return msg_id


class BaseCoSimulator(ABC):
    """Base class for co-simulation between robotics and network simulation."""
    
    def __init__(self, network_simulator: Any, robotics_simulator: Any, timestep: float = 0.1):
        """
        Initialize the co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            robotics_simulator: Instance of robotics simulator (e.g., gym, carla)
            timestep: Unified simulation timestep in seconds (default: 0.1)
        """
        self.network_simulator = network_simulator
        self.robotics_simulator = robotics_simulator
        
        # Set a single unified timestep for all simulation components
        self.timestep = timestep
        
        # Initialize time manager
        self._time_manager = TimeManager(timestep)
        
        # Initialize message handler
        self._message_handler = MessageHandler(network_simulator)
        
        # Define flow IDs for different message types (for backward compatibility)
        self.CLIENT_TO_SERVER_FLOW = MessageHandler.CLIENT_TO_SERVER_FLOW
        self.SERVER_TO_CLIENT_FLOW = MessageHandler.SERVER_TO_CLIENT_FLOW
        
        logger.info("BaseCoSimulator initialized with unified timestep: %f", self.timestep)
        
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
        self._time_manager.advance_time()
        
    def get_current_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns:
            float: Current simulation time in seconds
        """
        return self._time_manager.get_current_time()
    
    def get_timestep(self) -> float:
        """
        Get the simulation timestep.
        
        Returns:
            float: Unified simulation timestep in seconds
        """
        return self._time_manager.get_timestep()
    
    @property
    def current_time(self) -> float:
        """Property for backward compatibility."""
        return self._time_manager.get_current_time()
    
    @current_time.setter
    def current_time(self, value: float) -> None:
        """Setter for backward compatibility."""
        self._time_manager.current_time = value
    
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing simulators")
        self.robotics_simulator.close()
        self.network_simulator.close() 