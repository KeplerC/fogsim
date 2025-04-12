import numpy as np
from typing import Dict, Tuple, Any, List
from ..base import BaseCoSimulator

class GymCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Gym environments."""
    
    def __init__(self, network_simulator: Any, gym_env: Any, timestep: float = 0.1):
        """
        Initialize the Gym co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
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
        messages = self._process_network_messages()
        for message in messages:
            self._handle_message(message)

        # Calculate network latency for this step
        action_latency = 0.0
        observation_latency = 0.0
        
        # Simulate sending action from client to server
        client_send_time = self.get_current_time()
        action_msg_id = self._send_message(
            {'action': action, 'timestamp': client_send_time},
            flow_id=self.CLIENT_TO_SERVER_FLOW,  # This is 0
            size=self._estimate_message_size(action)
        )
        
        # Estimate latency for action packet
        action_size = self._estimate_message_size(action)
        action_latency = self.network_simulator.estimate_latency(
            size=action_size,
            flow_id=self.CLIENT_TO_SERVER_FLOW
        )
        
        # Step the robotics simulator (server side)
        result = self.robotics_simulator.step(action)
        
        # Handle both old and new Gym API formats
        if len(result) == 5:  # New Gymnasium API: obs, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old Gym API: obs, reward, done, info
            observation, reward, done, info = result
            
        self.current_observation = observation
        
        # Send observation through network simulator (server to client)
        if observation is not None:
            # Record send time
            server_send_time = self.get_current_time()
            
            # Send observation through network
            obs_msg_id = self._send_message(
                {'observation': observation, 'timestamp': server_send_time},
                flow_id=self.SERVER_TO_CLIENT_FLOW,  # This is 1
                size=self._estimate_message_size(observation)
            )
            
            # Estimate latency for observation packet
            observation_size = self._estimate_message_size(observation)
            observation_latency = self.network_simulator.estimate_latency(
                size=observation_size, 
                flow_id=self.SERVER_TO_CLIENT_FLOW
            )
        
        # Add network info to the info dict
        if not isinstance(info, dict):
            info = {}
        info['action_latency'] = action_latency
        info['observation_latency'] = observation_latency
        info['total_latency'] = action_latency + observation_latency
        
        # Advance simulation time
        self._advance_time()
            
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
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
    
    def _estimate_message_size(self, observation: np.ndarray) -> float:
        """
        Estimate the size of a message in bytes.
        
        Args:
            observation: Observation array
            
        Returns:
            float: Estimated size in bytes
        """
        if isinstance(observation, np.ndarray):
            # Numpy arrays have known memory usage
            return observation.nbytes
        elif isinstance(observation, dict):
            # For dictionaries, estimate based on keys
            total_size = 100  # Base size
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                else:
                    # Rough estimate for other types
                    total_size += 100
            return float(total_size)
        else:
            # Default size if we can't estimate
            return 1000.0
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.get_current_time() 