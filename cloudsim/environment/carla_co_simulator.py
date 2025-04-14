import numpy as np
import logging
import pickle
from typing import Dict, Tuple, Any, List, Optional
from ..base import BaseCoSimulator

# Set up logging
logger = logging.getLogger(__name__)

class CarlaCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Carla environments."""
    
    def __init__(self, network_simulator: Any, carla_env: Any, timestep: float = 0.1):
        """
        Initialize the Carla co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., ns3)
            carla_env: Carla environment instance
            timestep: Simulation timestep in seconds (default: 0.1)
        """
        super().__init__(network_simulator, carla_env, timestep)
        self.current_observation = None
        self.last_action = None
        self.received_action_this_step = False
        
        # Track observation IDs
        self.pending_observations = {}  # Maps observation_id -> observation data
        self.observation_counter = 0    # To generate unique observation IDs
        self.last_received_observation_id = None
        
        # Track latest network observation
        self.last_network_observation = None
        
        logger.info("CarlaCoSimulator initialized with Carla environment")
        
    def step(self, action: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the Carla simulator.
                   If None and no message arrives, the previous timestep's action will be used.
            
        Returns:
            np.ndarray: Observation received through the network
        """
        logger.info("Starting co-simulation step at time %f", self.get_current_time())
        
        # Process any network messages that should be visible now
        messages = self._process_network_messages()
        self.received_action_this_step = False
        
        # Process all messages to receive observations or actions
        for message in messages:
            if isinstance(message, dict):
                current_time = self.get_current_time()
                
                # Process observation message (client side received observation)
                if 'observation' in message and 'observation_id' in message:
                    observation_id = message['observation_id']
                    self.last_received_observation_id = observation_id
                    # Store this observation as our latest received through network
                    self.last_network_observation = message.get('observation')
                    logger.info(f"Received observation {observation_id}")
                
                # Process action message (server side received action)
                if 'action' in message and 'responding_to_observation' in message:
                    response_to_observation = message['responding_to_observation']
                    # Update the last_action based on the received message
                    self.last_action = message['action']
                    self.received_action_this_step = True
                    logger.info(f"Received action for observation {response_to_observation}: {str(self.last_action)}")
            
            # Handle the message
            self._handle_message(message)
                   
        # If client provided an action AND we have received an observation to respond to,
        # send that action through the network with observation context
        if action is not None and self.last_received_observation_id is not None:
            # Simulate sending action from client to server
            client_send_time = self.get_current_time()
            action_size = self._estimate_message_size(action)
            action_msg_id = self._send_message(
                {
                    'action': action, 
                    'timestamp': client_send_time,
                    'responding_to_observation': self.last_received_observation_id
                },
                flow_id=self.CLIENT_TO_SERVER_FLOW,
                size=action_size
            )
            logger.info(f"Sent action in response to observation {self.last_received_observation_id}")

        # Use the last action received from the network, or keep using the previous one
        if not self.received_action_this_step and self.last_action is None:
            # For Carla, we might need a default action if none exists
            # This will depend on the Carla environment's action space
            try:
                # Try to create a neutral action (no throttle, no steering)
                server_action = np.zeros(2)  # [throttle, steering]
                logger.info("No prior action - using neutral action: %s", str(server_action))
            except Exception as e:
                logger.error("Failed to create default action: %s", str(e))
                server_action = np.zeros(1)
            self.last_action = server_action
        else:
            # Use the last action we received
            server_action = self.last_action
            logger.info("Using %s action: %s", 
                       "newly received" if self.received_action_this_step else "previous", 
                       str(server_action))
        
        # Step the Carla simulator with the action
        logger.info("Stepping Carla simulator with action %s", str(server_action))
        observation = self.robotics_simulator.step(server_action)
        self.current_observation = observation
        
        # Send observation through network simulator (server to client)
        if observation is not None:
            # Generate a unique observation ID
            self.observation_counter += 1
            observation_id = self.observation_counter
            
            # Record send time
            server_send_time = self.get_current_time()
            
            # Store this observation in our pending observations map
            self.pending_observations[observation_id] = {
                'observation': observation,
                'timestamp': server_send_time,
                'responded_to': False
            }
            
            # Send observation through network with its ID
            observation_size = self._estimate_message_size(observation)
            obs_msg_id = self._send_message(
                {
                    'observation': observation, 
                    'timestamp': server_send_time,
                    'observation_id': observation_id
                },
                flow_id=self.SERVER_TO_CLIENT_FLOW,
                size=observation_size
            )
            logger.info(f"Sent observation with ID {observation_id}")
        
        # Advance simulation time
        self._advance_time()
        
        # Return the observation that came through the network (with delay),
        # or the direct observation if nothing has come through the network yet
        return self.last_network_observation if self.last_network_observation is not None else observation
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        logger.info("Resetting CarlaCoSimulator")
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
        self.current_time = 0.0
        self.last_action = None
        self.received_action_this_step = False
        # Reset observation tracking
        self.pending_observations = {}
        self.observation_counter = 0
        self.last_received_observation_id = None
        self.last_network_observation = None
        
        return observation
    
    def render(self, mode: str = 'human') -> Any:
        """Render the current state of the simulation."""
        logger.info("Rendering Carla environment")
        try:
            frame = self.robotics_simulator.render()
            return frame
        except Exception as e:
            logger.error("Render failed: %s", str(e))
            print(f"Render failed: {e}")
            return None
    
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

    def _estimate_message_size(self, data: Any) -> float:
        """
        Estimate the size of a message in bytes.
        
        Args:
            data: Data to estimate size of
            
        Returns:
            float: Estimated size in bytes
        """
        size = 0.0
        if isinstance(data, np.ndarray):
            size = float(data.nbytes)
        elif isinstance(data, dict):
            total_size = 100.0  # Base size
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                else:
                    total_size += 100.0
            size = total_size
        elif isinstance(data, list):
            size = float(len(data))
        else:
            try:
                # Use pickle to estimate the size
                size = float(len(pickle.dumps(data)))
            except Exception as e:
                logger.debug("Failed to pickle data: %s", str(e))
                size = 1000.0

        logger.info("Estimated message size: %f bytes", size)
        return size
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.get_current_time()
