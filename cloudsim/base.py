from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional, List
import numpy as np
import time

class BaseCoSimulator(ABC):
    """Base class for co-simulation between robotics and network simulation."""
    
    def __init__(self, network_simulator: Any, robotics_simulator: Any, timestep: float = 0.1):
        """
        Initialize the co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            robotics_simulator: Instance of robotics simulator (e.g., gym, carla)
            timestep: Simulation timestep in seconds (default: 0.1)
        """
        self.network_simulator = network_simulator
        self.robotics_simulator = robotics_simulator
        self.timestep = timestep
        self.current_time = timestep
        self._last_step_time: Optional[float] = None
        
        # Define flow IDs for different message types
        self.CLIENT_TO_SERVER_FLOW = 0
        self.SERVER_TO_CLIENT_FLOW = 1
        
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
        # Run the network simulator until current time
        self.network_simulator.run_until(self.current_time)
        
        # Get ready messages
        return self.network_simulator.get_ready_messages()
    
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
        return self.network_simulator.register_packet(message, flow_id, size)
    
    def _advance_time(self) -> None:
        """Advance the simulation time by one timestep."""
        self.current_time += self.timestep
        
    def get_current_time(self) -> float:
        """
        Get the current simulation time.
        
        Returns:
            float: Current simulation time in seconds
        """
        return self.current_time
    
    def get_timestep(self) -> float:
        """
        Get the simulation timestep.
        
        Returns:
            float: Simulation timestep in seconds
        """
        return self.timestep
    
    def close(self) -> None:
        """Clean up resources."""
        self.robotics_simulator.close()
        self.network_simulator.close() 