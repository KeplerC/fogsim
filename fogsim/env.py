"""Main environment interface for FogSim.

This module provides the main Env class that follows the Mujoco/Roboverse
interface pattern, integrating handlers with network simulation.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import logging

from .handlers.base_handler import BaseHandler
from .network.nspy_simulator import NSPyNetworkSimulator
from .network.config import NetworkConfig
from .time_backend import SimulationMode, UnifiedTimeManager, TimeSubscriber
from .message_passing import MessageBus, SimpleMessageHandler, TimedMessage
from .network_control import NetworkControlManager, NetworkConfig as NetworkControlConfig
from .real_network_client import RealNetworkClient, RealNetworkMessageHandler

logger = logging.getLogger(__name__)


class Env(TimeSubscriber):
    """Main FogSim environment class.
    
    This class provides a unified interface for co-simulation between robotics
    simulators and network simulation, following the Mujoco/Roboverse pattern.
    
    Args:
        handler: Handler instance for the robotics simulator
        network_config: Network simulation configuration
        enable_network: Whether to enable network simulation (default: True)
        timestep: Simulation timestep in seconds (default: 0.1)
        mode: Simulation mode (VIRTUAL, SIMULATED_NET, or REAL_NET)
    """
    
    def __init__(self,
                 handler: BaseHandler,
                 network_config: Optional[NetworkConfig] = None,
                 enable_network: bool = True,
                 timestep: float = 0.1,
                 mode: SimulationMode = SimulationMode.VIRTUAL,
                 real_network_host: str = "127.0.0.1",
                 real_network_port: int = 8765):
        """Initialize the FogSim environment."""
        self.handler = handler
        self.network_config = network_config or NetworkConfig()
        self.enable_network = enable_network
        self.timestep = timestep
        self.mode = mode
        self.real_network_host = real_network_host
        self.real_network_port = real_network_port
        
        # Initialize unified time manager
        self.time_manager = UnifiedTimeManager(mode, timestep)
        self.time_manager.register_subscriber(self)
        
        # Setup network simulator if enabled
        self.network_sim = None
        if self.enable_network:
            self.network_sim = self._setup_network()
        
        # Initialize message bus
        self.message_bus = MessageBus(self.time_manager, self.network_sim)
        
        # Initialize network control
        self.network_control = NetworkControlManager(mode, network_simulator=self.network_sim)
        
        # Setup message handlers
        if self.mode == SimulationMode.REAL_NET and self.enable_network:
            # Real network mode: use client that forwards to server
            self.real_network_client = RealNetworkClient(
                server_host=real_network_host,
                server_port=real_network_port,
                protocol="tcp"
            )
            self.real_network_client.start()
            
            self.robot_handler = RealNetworkMessageHandler(self.real_network_client, "robot")
            self.controller_handler = RealNetworkMessageHandler(self.real_network_client, "controller")
        else:
            # Virtual or simulated modes: use simple handlers
            self.real_network_client = None
            self.robot_handler = SimpleMessageHandler()
            self.controller_handler = SimpleMessageHandler()
            
        self.message_bus.register_handler("robot", self.robot_handler)
        self.message_bus.register_handler("controller", self.controller_handler)
        
        # Launch handler
        handler.launch()
        
        # Initialize state tracking
        self._step_count = 0
        self._episode_count = 0
        
        # Network message tracking
        self._pending_observation_id = 0
        self._last_observation_sent = None
        self._last_action_received = None
        self._network_latencies = []
        
        logger.info(f"FogSim environment initialized with {type(handler).__name__} in {mode.value} mode")
    
    def _setup_network(self) -> NSPyNetworkSimulator:
        """Setup network simulator based on configuration."""
        logger.info("Setting up network simulator with config: %s", self.network_config)
        
        # Create network simulator with configuration
        network_sim = NSPyNetworkSimulator(
            source_rate=self.network_config.source_rate,
            weights=self.network_config.flow_weights,
            debug=False
        )
        
        # TODO: Add support for different topologies, congestion control, etc.
        # based on network_config settings
        
        return network_sim
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Returns:
            Tuple of (observation, extra_info)
        """
        # Reset handler
        self.handler.set_states()
        states = self.handler.get_states()
        
        # Reset network simulator if enabled
        if self.network_sim is not None:
            self.network_sim.reset()
        
        # Reset tracking variables
        self.time_manager.reset()
        self._step_count = 0
        self._episode_count += 1
        self._pending_observation_id = 0
        self._last_observation_sent = None
        self._last_action_received = None
        self._network_latencies.clear()
        
        # Clear message handlers
        self.robot_handler.clear()
        self.controller_handler.clear()
        
        # Get observation and extra info
        observation = self._get_observation(states)
        extra_info = self.handler.get_extra()
        
        # Add environment metadata to extra info
        extra_info.update({
            'network_enabled': self.enable_network,
            'timestep': self.timestep,
            'episode': self._episode_count,
            'mode': self.mode.value
        })
        
        logger.info("Environment reset, episode %d", self._episode_count)
        
        return observation, extra_info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, bool, Dict[str, Any]]:
        """Step the environment forward.
        
        Args:
            action: Action to apply
            
        Returns:
            Tuple of (observation, reward, success, termination, timeout, extra_info)
        """
        # Handle network simulation if enabled
        if self.network_sim is not None:
            observation, reward, done, info = self._step_with_network(action)
        else:
            observation, reward, done, info = self._step_without_network(action)
        
        # Extract success, termination, and timeout from done/info
        success = info.get('success', False)
        termination = info.get('termination', done)
        timeout = info.get('timeout', False)
        
        # Add extra metadata
        extra_info = self.handler.get_extra()
        extra_info.update(info)
        extra_info.update({
            'step': self._step_count,
            'time': self.time_manager.now(),
            'network_latencies': self._network_latencies.copy()
        })
        
        return observation, reward, success, termination, timeout, extra_info
    
    def _step_without_network(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step without network simulation (direct control)."""
        # Apply action
        self.handler.set_states(action=action)
        
        # Step simulation
        self.handler.step()
        
        # Get new states
        states = self.handler.get_states()
        
        # Update time tracking
        self.time_manager.advance_time()
        self._step_count += 1
        
        # Get observation and compute reward
        observation = self._get_observation(states)
        reward = self._get_reward(states)
        done = self._get_termination(states) or self._get_timeout(states)
        
        info = {
            'success': self._get_success(states),
            'termination': self._get_termination(states),
            'timeout': self._get_timeout(states)
        }
        
        return observation, reward, done, info
    
    def _step_with_network(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Step with network simulation (delayed control)."""
        # Send current observation through network
        states = self.handler.get_states()
        observation = self._get_observation(states)
        current_time = self.time_manager.now()
        
        # Create observation message
        self._pending_observation_id += 1
        obs_message = {
            'observation': observation,
            'observation_id': self._pending_observation_id,
            'timestamp': current_time,
            'states': states
        }
        
        # Send observation from robot to controller
        obs_delay = self._calculate_network_delay(observation)
        self.message_bus.send(
            sender_id="robot",
            receiver_id="controller",
            payload=obs_message,
            delay=obs_delay,
            message_type="observation"
        )
        self._last_observation_sent = obs_message
        
        # Send action from controller to robot
        action_message = {
            'action': action,
            'responding_to_observation': self._pending_observation_id - 1,
            'timestamp': current_time
        }
        
        action_delay = self._calculate_network_delay(action)
        self.message_bus.send(
            sender_id="controller",
            receiver_id="robot",
            payload=action_message,
            delay=action_delay,
            message_type="action"
        )
        
        # Advance time
        self.time_manager.advance_time()
        self._step_count += 1
        
        # Process received messages
        messages = self.robot_handler.received_messages + self.controller_handler.received_messages
        
        # Default values if no messages received
        received_action = None  # Don't use current action as fallback
        network_observation = observation
        network_states = states
        
        for msg in messages:
            # Extract payload from TimedMessage
            payload = msg.payload if hasattr(msg, 'payload') else msg
            
            if isinstance(payload, dict) and 'action' in payload:
                # Received action from network
                received_action = payload['action']
                self._last_action_received = payload
                
                # Calculate action latency
                action_latency = self.time_manager.now() - payload['timestamp']
                self._network_latencies.append({
                    'type': 'action',
                    'latency': action_latency,
                    'time': self.time_manager.now()
                })
                
            elif isinstance(payload, dict) and 'observation' in payload:
                # Received observation from network
                network_observation = payload['observation']
                network_states = payload.get('states', states)
                
                # Calculate observation latency
                obs_latency = self.time_manager.now() - payload['timestamp']
                self._network_latencies.append({
                    'type': 'observation',
                    'latency': obs_latency,
                    'time': self.time_manager.now()
                })
        
        # Apply the received action (which may be delayed)
        # If no action received yet, use the last received action or neutral action
        if received_action is None:
            if hasattr(self, '_last_received_action'):
                received_action = self._last_received_action
            else:
                # Use neutral action for CartPole (action 0)
                received_action = 0
        else:
            # Store the last received action
            self._last_received_action = received_action
            
        self.handler.set_states(states={}, action=received_action)
        self.handler.step()
        
        # Get reward and termination based on current state
        current_states = self.handler.get_states()
        reward = self._get_reward(current_states)
        done = self._get_termination(current_states) or self._get_timeout(current_states)
        
        info = {
            'success': self._get_success(current_states),
            'termination': self._get_termination(current_states),
            'timeout': self._get_timeout(current_states),
            'network_delay': len(messages) > 0,
            'num_messages_received': len(messages)
        }
        
        # Add real network statistics if available
        if self.real_network_client:
            network_stats = self.real_network_client.get_stats()
            info['real_network_stats'] = network_stats
            # Update network latencies with actual measured values
            if network_stats['average_latency_ms'] > 0:
                self._network_latencies.append({
                    'type': 'real_network',
                    'latency': network_stats['average_latency_ms'] / 1000.0,  # Convert to seconds
                    'time': self.time_manager.now()
                })
        
        # Return the network-delayed observation (what the controller sees)
        return network_observation, reward, done, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state.
        
        Returns:
            Rendered frame or None
        """
        return self.handler.render()
    
    def sync_to_time(self, time: float) -> None:
        """TimeSubscriber interface - called when time is synchronized."""
        # Clear processed messages from handlers
        self.robot_handler.clear()
        self.controller_handler.clear()
    
    def configure_network(self, config: NetworkControlConfig) -> None:
        """Configure network parameters."""
        self.network_control.configure(config)
    
    def close(self) -> None:
        """Close the environment and clean up resources."""
        logger.info("Closing FogSim environment")
        
        # Unregister from time manager
        self.time_manager.unregister_subscriber(self)
        
        # Reset network control
        self.network_control.reset()
        
        # Close real network client if used
        if self.real_network_client:
            self.real_network_client.stop()
        
        # Close handler
        self.handler.close()
        
        # Close network simulator if enabled
        if self.network_sim is not None:
            self.network_sim.close()
    
    def _get_observation(self, states: Dict[str, Any]) -> np.ndarray:
        """Extract observation from states.
        
        Args:
            states: State dictionary from handler
            
        Returns:
            Observation array
        """
        # Default implementation: return 'observation' key if available
        if 'observation' in states and states['observation'] is not None:
            obs = states['observation']
            # Ensure observation is a numpy array
            if not isinstance(obs, np.ndarray):
                obs = np.array(obs)
            return obs
        
        # Fallback: concatenate qpos and qvel for Mujoco-style envs
        if 'qpos' in states and 'qvel' in states:
            return np.concatenate([states['qpos'], states['qvel']])
        
        # Last resort: return empty array
        logger.warning("No observation found in states")
        return np.array([])
    
    def _get_reward(self, states: Dict[str, Any]) -> float:
        """Compute reward from states.
        
        Args:
            states: State dictionary from handler
            
        Returns:
            Reward value
            
        Note: This is a placeholder. Real implementations should override
        this method or use a reward function.
        """
        return states.get('reward', 0.0)
    
    def _get_success(self, states: Dict[str, Any]) -> bool:
        """Check if task is successful.
        
        Args:
            states: State dictionary from handler
            
        Returns:
            Success flag
        """
        return states.get('success', False)
    
    def _get_termination(self, states: Dict[str, Any]) -> bool:
        """Check if episode should terminate.
        
        Args:
            states: State dictionary from handler
            
        Returns:
            Termination flag
        """
        return states.get('done', False) or states.get('termination', False)
    
    def _get_timeout(self, states: Dict[str, Any]) -> bool:
        """Check if episode has timed out.
        
        Args:
            states: State dictionary from handler
            
        Returns:
            Timeout flag
        """
        # Default timeout after 1000 steps
        max_steps = states.get('max_steps', 1000)
        return self._step_count >= max_steps
    
    def _estimate_message_size(self, data: Any) -> float:
        """Estimate size of message in bytes.
        
        Args:
            data: Data to estimate size for
            
        Returns:
            Estimated size in bytes
        """
        if isinstance(data, np.ndarray):
            return float(data.nbytes)
        elif isinstance(data, (list, tuple)):
            return len(data) * 100.0  # Rough estimate
        else:
            return 1000.0  # Default size
    
    def _calculate_network_delay(self, data: Any) -> float:
        """Calculate network delay based on mode and data size."""
        if self.mode == SimulationMode.VIRTUAL:
            # Virtual mode: use configured delays
            return 0.01  # 10ms default
        elif self.mode == SimulationMode.SIMULATED_NET:
            # Let network simulator calculate delay
            size = self._estimate_message_size(data)
            # Simple model: 10ms base + size-dependent delay
            return 0.01 + (size / 1e7)  # 10MB/s
        else:  # REAL_NET
            # Real network: no artificial delay
            return 0.0