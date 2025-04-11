import numpy as np
from typing import Dict, Tuple, Any
from ..base import BaseCoSimulator

class GymCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Gym environments."""
    
    def __init__(self, network_simulator: Any, gym_env: Any, timestep: float = 0.1):
        """
        Initialize the Gym co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., ns3)
            gym_env: Gym environment instance
            timestep: Simulation timestep in seconds (default: 0.1)
        """
        super().__init__(network_simulator, gym_env, timestep)
        self.current_observation = None
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator
            
        Returns:
            Tuple containing observation, reward, done, and info
        """
        # Process any network messages that should be visible now
        self._process_network_messages()

        print(f"Current time: {self.get_current_time()}")
        print("action: ", action)
        
        # Step the robotics simulator
        result = self.robotics_simulator.step(action)
        
        # Handle both old and new Gym API formats
        if len(result) == 5:  # New Gymnasium API: obs, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old Gym API: obs, reward, done, info
            observation, reward, done, info = result
            
        self.current_observation = observation
        
        # Send observation through network simulator
        if observation is not None:
            self._send_observation(observation)
        
        # Advance simulation time
        self._advance_time()
            
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
        self.scheduled_messages.clear()
        self.current_time = 0.0
        return observation
    
    def render(self, mode: str = 'human') -> None:
        """Render the current state of the simulation."""
        try:
            # New Gymnasium API (may not accept mode parameter)
            frame = self.robotics_simulator.render()
            return frame
        except TypeError:
            try:
                # Old Gym API (requires mode parameter)
                frame = self.robotics_simulator.render(mode=mode)
                return frame
            except Exception as e:
                print(f"Render failed: {e}")
    
    def _handle_message(self, message: Any) -> None:
        """
        Handle a received message from the network simulator.
        
        Args:
            message: The received message
        """
        # Update the current observation with the received message
        if isinstance(message, dict) and 'observation' in message:
            self.current_observation = message['observation']
    
    def _send_observation(self, observation: np.ndarray) -> None:
        """
        Send observation through the network simulator.
        
        Args:
            observation: Current observation to send
        """
        # Get network latency from ns3
        latency = self.network_simulator.get_latency()
        
        # Schedule the message to be processed after the latency
        message = {
            'observation': observation,
            'timestamp': self.get_current_time()
        }
        self._schedule_message(message, latency)
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.robotics_simulator.get_time() 