import numpy as np
import logging
from typing import Dict, Tuple, Any, List, Optional
from ..base import BaseCoSimulator

# Set up logging
logger = logging.getLogger(__name__)

class GymCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Gym environments."""
    
    def __init__(self, network_simulator: Any, gym_env: Any, cosim_timestep: float,
                 network_timestep: Optional[float] = None, robotics_timestep: Optional[float] = None):
        """
        Initialize the Gym co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            gym_env: Gym environment instance
            cosim_timestep: Co-simulation timestep in seconds
            network_timestep: Network simulator timestep in seconds (default: same as cosim_timestep)
            robotics_timestep: Robotics simulator timestep in seconds (default: same as cosim_timestep)
        """
        super().__init__(
            network_simulator=network_simulator, 
            robotics_simulator=gym_env, 
            cosim_timestep=cosim_timestep,
            network_timestep=network_timestep,
            robotics_timestep=robotics_timestep
        )
        self.current_observation = None
        logger.info("GymCoSimulator initialized with gym environment %s", type(gym_env).__name__)
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator
            
        Returns:
            Tuple containing observation, reward, done, and info
        """
        logger.info("Starting co-simulation step at time %f with action %s", 
                    self.get_current_time(), str(action))
                    
        # Process any network messages that should be visible now
        messages = self._process_network_messages()
        for message in messages:
            self._handle_message(message)
            logger.info("Handled message: %s", str(message)[:100] + "..." if len(str(message)) > 100 else str(message))

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
        logger.info("Sent action from client to server with msg_id=%s at time=%f", 
                    action_msg_id, client_send_time)
        
        # Estimate latency for action packet
        action_size = self._estimate_message_size(action)
        action_latency = self.network_simulator.estimate_latency(
            size=action_size,
            flow_id=self.CLIENT_TO_SERVER_FLOW
        )
        logger.info("Estimated action latency: %f seconds", action_latency)
        
        # Step the robotics simulator (server side)
        logger.info("Stepping robotics simulator with action %s", str(action))
        result = self.robotics_simulator.step(action)
        
        # Handle both old and new Gym API formats
        if len(result) == 5:  # New Gymnasium API: obs, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = result
            done = terminated or truncated
            logger.info("Gym step result (new API): reward=%f, terminated=%s, truncated=%s", 
                        reward, terminated, truncated)
        else:  # Old Gym API: obs, reward, done, info
            observation, reward, done, info = result
            logger.info("Gym step result (old API): reward=%f, done=%s", reward, done)
            
        self.current_observation = observation
        
        # Send observation through network simulator (server to client)
        if observation is not None:
            # Record send time
            server_send_time = self.get_current_time()
            
            # Send observation through network
            observation_size = self._estimate_message_size(observation)
            obs_msg_id = self._send_message(
                {'observation': observation, 'timestamp': server_send_time},
                flow_id=self.SERVER_TO_CLIENT_FLOW,  # This is 1
                size=observation_size
            )
            logger.info("Sent observation from server to client with msg_id=%s at time=%f, size=%f", 
                        obs_msg_id, server_send_time, observation_size)
            
            # Estimate latency for observation packet
            observation_latency = self.network_simulator.estimate_latency(
                size=observation_size, 
                flow_id=self.SERVER_TO_CLIENT_FLOW
            )
            logger.info("Estimated observation latency: %f seconds", observation_latency)
        else:
            logger.warning("Received None observation from robotics simulator")
        
        # Add network info to the info dict
        if not isinstance(info, dict):
            info = {}
        info['action_latency'] = action_latency
        info['observation_latency'] = observation_latency
        info['total_latency'] = action_latency + observation_latency
        
        logger.info("Network latencies - action: %f, observation: %f, total: %f", 
                    action_latency, observation_latency, action_latency + observation_latency)
        
        # Advance simulation time
        self._advance_time()
            
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        logger.info("Resetting GymCoSimulator")
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
        self.current_time = 0.0
        logger.info("Reset complete, initial observation shape: %s", 
                    str(observation.shape) if hasattr(observation, 'shape') else str(type(observation)))
        return observation
    
    def render(self, mode: str = 'human') -> None:
        """Render the current state of the simulation."""
        logger.info("Rendering gym environment with mode=%s", mode)
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
                logger.error("Render failed: %s", str(e))
                print(f"Render failed: {e}")
    
    def _handle_message(self, message: Any) -> None:
        """
        Handle a received message from the network simulator.
        
        Args:
            message: The received message
        """
        logger.info("Handling message: %s", str(message)[:50] + "..." if isinstance(message, dict) and len(str(message)) > 50 else str(message))
        # Update the current observation with the received message
        if isinstance(message, dict) and 'observation' in message:
            self.current_observation = message['observation']
            logger.info("Updated current observation from network message")
    
    def _estimate_message_size(self, observation: np.ndarray) -> float:
        """
        Estimate the size of a message in bytes.
        
        Args:
            observation: Observation array
            
        Returns:
            float: Estimated size in bytes
        """
        size = 0.0
        if isinstance(observation, np.ndarray):
            # Numpy arrays have known memory usage
            size = float(observation.nbytes)
        elif isinstance(observation, dict):
            # For dictionaries, estimate based on keys
            total_size = 100.0  # Base size
            for key, value in observation.items():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                else:
                    # Rough estimate for other types
                    total_size += 100.0
            size = total_size
        else:
            # Default size if we can't estimate
            size = 1000.0
            
        logger.info("Estimated message size: %f bytes", size)
        return size
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.get_current_time() 