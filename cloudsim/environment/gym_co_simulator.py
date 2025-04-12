import numpy as np
import logging
from typing import Dict, Tuple, Any, List, Optional
from ..base import BaseCoSimulator
import pickle
# Set up logging
logger = logging.getLogger(__name__)

class GymCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Gym environments."""
    
    def __init__(self, network_simulator: Any, gym_env: Any, timestep: float):
        """
        Initialize the Gym co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            gym_env: Gym environment instance
            timestep: Unified simulation timestep in seconds
        """
        super().__init__(
            network_simulator=network_simulator, 
            robotics_simulator=gym_env, 
            timestep=timestep
        )
        self.current_observation = None
        self.last_action = None
        self.received_action_this_step = False
        logger.info("GymCoSimulator initialized with gym environment %s", type(gym_env).__name__)
        
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator.
                   If None and no message arrives, the previous timestep's action will be used.
            
        Returns:
            Tuple containing observation, reward, done, and info
        """
        logger.info("Starting co-simulation step at time %f", self.get_current_time())
                   
        # Process any network messages that should be visible now
        messages = self._process_network_messages()
        self.received_action_this_step = False
        
        # Process all messages
        for message in messages:
            self._handle_message(message)
            logger.info("Handled message: %s", str(message)[:100] + "..." if len(str(message)) > 100 else str(message))

        # If no action was provided by the user or received from the network
        if action is None and not self.received_action_this_step:
            # Use the previous timestep's action if available
            if self.last_action is not None:
                action = self.last_action
                logger.info("No new action - persisting previous action: %s", str(action))
            else:
                # If no prior action exists, sample a random one
                try:
                    if hasattr(self.robotics_simulator, 'action_space'):
                        action = self.robotics_simulator.action_space.sample()
                        logger.info("No prior action - sampling random action: %s", str(action))
                    else:
                        # Fallback to zeros if we can't sample
                        action = np.zeros(1)
                        logger.info("No action space found - using zero action: %s", str(action))
                except Exception as e:
                    logger.error("Failed to sample action: %s", str(e))
                    action = np.zeros(1)
        elif action is None and self.received_action_this_step:
            # We received an action from the network during this step
            action = self.last_action
            logger.info("Using action received from network: %s", str(action))
        
        # Store this action as the last action for the next step
        self.last_action = action

        # Calculate network latency for this step
        action_latency = 0.0
        observation_latency = 0.0
        
        # Simulate sending action from client to server
        client_send_time = self.get_current_time()
        action_msg_id = self._send_message(
            {'action': action, 'timestamp': client_send_time},
            flow_id=self.CLIENT_TO_SERVER_FLOW,
            size=self._estimate_message_size(action)
        )
        
        # Estimate latency for action packet
        action_size = self._estimate_message_size(action)
        action_latency = self.network_simulator.estimate_latency(
            size=action_size,
            flow_id=self.CLIENT_TO_SERVER_FLOW
        )
        
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
                flow_id=self.SERVER_TO_CLIENT_FLOW,
                size=observation_size
            )
            
            # Estimate latency for observation packet
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
        info['action_from_previous_step'] = not self.received_action_this_step
        
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
        self.last_action = None
        self.received_action_this_step = False
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
        # Update the current observation with the received message
        if isinstance(message, dict):
            if 'observation' in message:
                self.current_observation = message['observation']
            
            # If the message contains an action, update our last_action
            if 'action' in message:
                self.last_action = message['action']
                self.received_action_this_step = True
                logger.info("Received action from network: %s", str(self.last_action))

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
        elif isinstance(observation, list):
            # For lists, estimate based on number of elements
            size = float(len(observation))
        else:
            # Default size if we can't estimate
            # pickle it     
            size = float(len(pickle.dumps(observation)))

        logger.info("Estimated message size: %f bytes", size)
        return size
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.get_current_time() 