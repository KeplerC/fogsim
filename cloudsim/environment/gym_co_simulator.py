import numpy as np
import logging
from typing import Dict, Tuple, Any, List, Optional
from ..base import BaseCoSimulator
import pickle
# Set up logging
logger = logging.getLogger(__name__)


class ObservationTracker:
    """Manages observation tracking and ID generation."""
    
    def __init__(self):
        self.pending_observations = {}  # Maps observation_id -> observation data
        self.observation_counter = 0    # To generate unique observation IDs
        self.last_received_observation_id = None  # Track which observation we're responding to
    
    def create_observation_id(self) -> int:
        """Generate a new unique observation ID."""
        self.observation_counter += 1
        return self.observation_counter
    
    def add_pending_observation(self, observation_id: int, observation_data: Dict[str, Any]) -> None:
        """Add a pending observation."""
        self.pending_observations[observation_id] = observation_data
    
    def update_last_received(self, observation_id: int) -> None:
        """Update the last received observation ID."""
        self.last_received_observation_id = observation_id
    
    def reset(self) -> None:
        """Reset tracking state."""
        self.pending_observations.clear()
        self.observation_counter = 0
        self.last_received_observation_id = None


class NetworkObservationState:
    """Manages network observation state."""
    
    def __init__(self):
        self.last_network_observation = None  # The most recent observation received through the network
        self.last_network_reward = 0.0  # The reward associated with the last network observation
        self.last_network_done = False  # The done flag associated with the last network observation
        self.last_network_info = {}     # The info associated with the last network observation
    
    def update(self, observation: Any, reward: float, done: bool, info: Dict) -> None:
        """Update the network observation state."""
        self.last_network_observation = observation
        self.last_network_reward = reward
        self.last_network_done = done
        self.last_network_info = info
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.last_network_observation = None
        self.last_network_reward = 0.0
        self.last_network_done = False
        self.last_network_info = {}


class MessageSizeEstimator:
    """Estimates message sizes for network transmission."""
    
    @staticmethod
    def estimate(data: Any) -> float:
        """Estimate the size of data in bytes."""
        if isinstance(data, np.ndarray):
            return float(data.nbytes)
        elif isinstance(data, dict):
            total_size = 100.0  # Base size
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    total_size += value.nbytes
                else:
                    total_size += 100.0
            return total_size
        elif isinstance(data, list):
            return float(len(data) * 100)  # Rough estimate
        else:
            # Default: pickle and measure
            return float(len(pickle.dumps(data)))

class GymCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Gym environments."""
    
    def __init__(self, network_simulator: Any, gym_env: Any, timestep: float,
                 observation_tracker: Optional[ObservationTracker] = None,
                 network_state: Optional[NetworkObservationState] = None,
                 size_estimator: Optional[MessageSizeEstimator] = None):
        """
        Initialize the Gym co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., NSPyNetworkSimulator)
            gym_env: Gym environment instance
            timestep: Unified simulation timestep in seconds
            observation_tracker: Optional observation tracker for dependency injection
            network_state: Optional network state manager for dependency injection
            size_estimator: Optional message size estimator for dependency injection
        """
        super().__init__(
            network_simulator=network_simulator, 
            robotics_simulator=gym_env, 
            timestep=timestep
        )
        self.current_observation = None
        self.last_action = None
        self.received_action_this_step = False
        
        # Use injected dependencies or create defaults
        self.observation_tracker = observation_tracker or ObservationTracker()
        self.network_state = network_state or NetworkObservationState()
        self.size_estimator = size_estimator or MessageSizeEstimator()
        
        logger.info("GymCoSimulator initialized with gym environment %s", type(gym_env).__name__)
        
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator.
                   If None and no message arrives, the previous timestep's action will be used.
            
        Returns:
            Tuple containing observation, reward, done, and info as received through the network
        """
        logger.info("Starting co-simulation step at time %f", self.get_current_time())
        
        # Track latency information for info dict
        action_latencies = []
        observation_latencies = []
        
        # Process any network messages that should be visible now
        messages = self._process_network_messages()
        self.received_action_this_step = False
        received_observation_this_step = False
        
        # Track round-trip latency when action arrives
        round_trip_latency = None
        
        # Variables to hold the latest network observation data
        network_observation = None
        network_reward = None
        network_done = None
        network_info = None
        
        # Process all messages to receive observations or actions
        for message in messages:
            if isinstance(message, dict):
                current_time = self.get_current_time()
                
                # Process observation message (client side received observation)
                if 'observation' in message and 'observation_id' in message:
                    observation_id = message['observation_id']
                    self.observation_tracker.update_last_received(observation_id)
                    
                    # Store this observation as our latest received through network
                    network_observation = message.get('observation')
                    network_reward = message.get('reward', 0.0)
                    network_done = message.get('done', False)
                    network_info = message.get('info', {})
                    received_observation_this_step = True
                    
                    # Calculate observation latency
                    if 'timestamp' in message:
                        latency = current_time - message['timestamp']
                        observation_latencies.append(latency)
                        logger.info(f"Received observation {observation_id} with latency {latency}")
                
                # Process action message (server side received action)
                if 'action' in message and 'responding_to_observation' in message:
                    response_to_observation = message['responding_to_observation']
                    
                    # Calculate action latency
                    if 'timestamp' in message:
                        latency = current_time - message['timestamp']
                        action_latencies.append(latency)
                        
                        # Only count round-trip for actions matching the observation they respond to
                        if response_to_observation in self.observation_tracker.pending_observations:
                            obs_timestamp = self.observation_tracker.pending_observations[response_to_observation].get('timestamp')
                            if obs_timestamp:
                                round_trip_latency = current_time - obs_timestamp
                                logger.info(f"Round-trip latency for observation {response_to_observation}: {round_trip_latency}")
                    
                    # Update the last_action based on the received message
                    self.last_action = message['action']
                    self.received_action_this_step = True
                    logger.info(f"Received action for observation {response_to_observation}: {str(self.last_action)}")
            
            # Handle the message (legacy method)
            self._handle_message(message)
        
        # If we received a new observation through the network, update our stored values
        if received_observation_this_step and network_observation is not None:
            self.network_state.update(network_observation, network_reward, network_done, network_info)
                   
        # If client provided an action AND we have received an observation to respond to,
        # send that action through the network with observation context
        if action is not None and self.observation_tracker.last_received_observation_id is not None:
            # Simulate sending action from client to server
            client_send_time = self.get_current_time()
            action_size = self.size_estimator.estimate(action)
            action_msg_id = self._send_message(
                {
                    'action': action, 
                    'timestamp': client_send_time,
                    'responding_to_observation': self.observation_tracker.last_received_observation_id
                },
                flow_id=self.CLIENT_TO_SERVER_FLOW,
                size=action_size
            )
            logger.info(f"Sent action in response to observation {self.observation_tracker.last_received_observation_id}")

        # Use the last action received from the network, or keep using the previous one
        if not self.received_action_this_step and self.last_action is None:
            # If no prior action exists, sample a random one
            try:
                if hasattr(self.robotics_simulator, 'action_space'):
                    server_action = self.robotics_simulator.action_space.sample()
                    logger.info("No prior action - sampling random action: %s", str(server_action))
                else:
                    # Fallback to zeros if we can't sample
                    server_action = np.zeros(1)
                    logger.info("No action space found - using zero action: %s", str(server_action))
            except Exception as e:
                logger.error("Failed to sample action: %s", str(e))
                server_action = np.zeros(1)
            self.last_action = server_action
        else:
            # Use the last action we received (either in this step or previous steps)
            server_action = self.last_action
            logger.info("Using %s action: %s", 
                       "newly received" if self.received_action_this_step else "previous", 
                       str(server_action))
        
        # Step the robotics simulator (server side) with the action that has gone through the network
        logger.info("Stepping robotics simulator with action %s", str(server_action))
        result = self.robotics_simulator.step(server_action)
        
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
            # Generate a unique observation ID
            observation_id = self.observation_tracker.create_observation_id()
            
            # Record send time
            server_send_time = self.get_current_time()
            
            # Store this observation in our pending observations map
            self.observation_tracker.add_pending_observation(observation_id, {
                'observation': observation,
                'timestamp': server_send_time,
                'responded_to': False,
                'reward': reward,
                'done': done,
                'info': info
            })
            
            # Send observation through network with its ID and additional data
            observation_size = self.size_estimator.estimate(observation)
            obs_msg_id = self._send_message(
                {
                    'observation': observation, 
                    'timestamp': server_send_time,
                    'observation_id': observation_id,
                    'reward': reward,
                    'done': done,
                    'info': info
                },
                flow_id=self.SERVER_TO_CLIENT_FLOW,
                size=observation_size
            )
            logger.info(f"Sent observation with ID {observation_id}")
        
        # Use network-received info or initialize if none exists
        network_info_to_return = self.network_state.last_network_info.copy() if self.network_state.last_network_info else {}
        if not isinstance(network_info_to_return, dict):
            network_info_to_return = {}
        
        # Include latency information in info dict
        if action_latencies:
            network_info_to_return['action_latencies'] = action_latencies
        if observation_latencies:
            network_info_to_return['observation_latencies'] = observation_latencies
            
        # Include round-trip latency if an action arrived this step
        if self.received_action_this_step and round_trip_latency is not None:
            network_info_to_return['round_trip_latency'] = round_trip_latency
        
        # Advance simulation time
        self._advance_time()
            
        # Return the observation that came through the network (with delay),
        # NOT the immediate result from the robotics simulator
        return (
            self.network_state.last_network_observation if self.network_state.last_network_observation is not None else observation,
            self.network_state.last_network_reward,
            self.network_state.last_network_done,
            network_info_to_return
        )
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        logger.info("Resetting GymCoSimulator")
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
        self._time_manager.reset_time()
        self.last_action = None
        self.received_action_this_step = False
        
        # Reset tracking components
        self.observation_tracker.reset()
        self.network_state.reset()
        
        # For the first observation after reset, we'll use the immediate observation
        # since there wouldn't be any network-received observation yet
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

    def _estimate_message_size(self, data: Any) -> float:
        """
        Estimate the size of a message in bytes.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            float: Estimated size in bytes
        """
        size = self.size_estimator.estimate(data)
        logger.info("Estimated message size: %f bytes", size)
        return size
    
    def get_time(self) -> float:
        """Get current simulation time."""
        return self.get_current_time() 