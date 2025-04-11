from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, Optional
import numpy as np
import time

class BaseCoSimulator(ABC):
    """Base class for co-simulation between robotics and network simulation."""
    
    def __init__(self, network_simulator: Any, robotics_simulator: Any, timestep: float = 0.1):
        """
        Initialize the co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., ns3)
            robotics_simulator: Instance of robotics simulator (e.g., gym, carla)
            timestep: Simulation timestep in seconds (default: 0.1)
        """
        self.network_simulator = network_simulator
        self.robotics_simulator = robotics_simulator
        self.scheduled_messages = {}  # Dictionary to store scheduled messages
        self.timestep = timestep
        self.current_time = 0.0
        self._last_step_time: Optional[float] = None
        
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
    
    def _process_network_messages(self) -> None:
        """Process any messages that should be visible at the current timestep."""
        messages_to_process = []
        
        # Find messages that should be processed at current time
        for msg_id, (msg, scheduled_time) in self.scheduled_messages.items():
            if scheduled_time <= self.current_time:
                messages_to_process.append((msg_id, msg))
        
        # Process messages and remove them from schedule
        for msg_id, msg in messages_to_process:
            self._handle_message(msg)
            del self.scheduled_messages[msg_id]
    
    @abstractmethod
    def _handle_message(self, message: Any) -> None:
        """
        Handle a received message from the network simulator.
        
        Args:
            message: The received message
        """
        pass
    
    def _schedule_message(self, message: Any, delay: float) -> None:
        """
        Schedule a message to be processed after a delay.
        
        Args:
            message: Message to schedule
            delay: Delay in seconds
        """
        scheduled_time = self.current_time + delay
        msg_id = id(message)  # Use message ID as key
        self.scheduled_messages[msg_id] = (message, scheduled_time)
    
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