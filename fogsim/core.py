"""
FogSim Core - Simplified co-simulation framework
"""

import logging
from enum import Enum
from typing import Tuple, Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Three FogSim simulation modes."""
    VIRTUAL = "virtual"           # Mode 1: Virtual timeline
    SIMULATED_NET = "simulated"   # Mode 2: Real clock + simulated network  
    REAL_NET = "real"            # Mode 3: Real clock + real network


class FogSim:
    """
    Simplified FogSim co-simulation framework.
    
    Implements the three modes described in CLAUDE.md:
    1. Virtual timeline (VIRTUAL)
    2. Real clock + simulated network (SIMULATED_NET) 
    3. Real clock + real network (REAL_NET)
    """
    
    def __init__(self, handler, mode: SimulationMode = SimulationMode.VIRTUAL, 
                 timestep: float = 0.1, network_config=None):
        """
        Initialize FogSim.
        
        Args:
            handler: Environment handler (GymHandler, CarlaHandler, etc.)
            mode: Simulation mode
            timestep: Simulation timestep in seconds
            network_config: Network configuration (optional)
        """
        self.handler = handler
        self.mode = mode
        self.timestep = timestep
        self.network_config = network_config
        
        # Initialize clock based on mode
        if mode == SimulationMode.VIRTUAL:
            from .clock.virtual_clock import VirtualClock
            self.clock = VirtualClock(timestep)
        else:
            from .clock.real_clock import RealClock
            self.clock = RealClock(timestep)
        
        # Initialize network based on mode
        self.network = self._init_network()
        
        # State tracking
        self.current_obs = None
        self.episode_step = 0
        
        # Launch handler
        self.handler.launch()
        
        logger.info(f"FogSim initialized: mode={mode.value}, timestep={timestep}")
    
    def _init_network(self):
        """Initialize network component based on simulation mode."""
        if self.mode == SimulationMode.VIRTUAL:
            from .network.nspy_simulator import NSPyNetworkSimulator
            # Use network config if provided
            if self.network_config:
                source_rate = getattr(self.network_config, 'source_rate', 1e9)  # Default to high bandwidth
                link_delay = getattr(self.network_config.topology, 'link_delay', 0.0) if hasattr(self.network_config, 'topology') else 0.0
                flow_weights = getattr(self.network_config, 'flow_weights', [1, 1])
                debug = getattr(self.network_config, 'debug', False)
                return NSPyNetworkSimulator(source_rate=source_rate, weights=flow_weights, debug=debug, link_delay=link_delay)
            else:
                return NSPyNetworkSimulator(source_rate=1e9)
        elif self.mode == SimulationMode.SIMULATED_NET:
            from .network.wallclock_simulator import WallclockNetworkSimulator
            # Use network config if provided
            if self.network_config:
                source_rate = getattr(self.network_config, 'source_rate', 1e9)  # Default to high bandwidth
                link_delay = getattr(self.network_config.topology, 'link_delay', 0.0) if hasattr(self.network_config, 'topology') else 0.0
                flow_weights = getattr(self.network_config, 'flow_weights', [1, 1])
                debug = getattr(self.network_config, 'debug', False)
                return WallclockNetworkSimulator(source_rate=source_rate, weights=flow_weights, debug=debug, link_delay=link_delay)
            else:
                return WallclockNetworkSimulator(source_rate=1e9)
        else:  # REAL_NET
            from .network.real_network import RealNetworkTransport
            if self.network_config:
                return RealNetworkTransport(self.network_config)
            else:
                return RealNetworkTransport()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step with proper network delays.
        
        The flow accounts for the fact that messages sent at time T arrive at time T+delay:
        1. Advance time 
        2. Run network simulation to current time to deliver any messages
        3. Get delayed actions/observations that have arrived
        4. Send NEW action through network (will arrive later)
        5. Execute handler with delayed or buffered action
        6. Send NEW observation through network (will arrive later)
        7. Return delayed or fresh observation
        
        Args:
            action: Action to execute
            
        Returns:
            observation, reward, success, termination, timeout, info
        """
        # Step 1: Advance time
        self.clock.advance()
        new_time = self.clock.now()
        
        # Step 2: Process network to get delayed messages that have arrived
        arrived_actions = []
        arrived_observations = []
        if self.network:
            # Run network simulation up to current time
            if hasattr(self.network, 'run_until'):
                self.network.run_until(new_time)
            
            # Get messages that have arrived
            if hasattr(self.network, 'get_ready_messages'):
                messages = self.network.get_ready_messages()
            else:
                messages = []
            
            # DEBUG: Log packet processing
            if self.episode_step % 10 == 0 or messages:
                pending = len(self.network.packet_tracker.pending_packets) if hasattr(self.network, 'packet_tracker') else 0
                logger.info(f"Step {self.episode_step}: Network time={new_time:.3f}, messages_arrived={len(messages)}, pending_packets={pending}")
            
            # Sort messages by type
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get('type') == 'action':
                        arrived_actions.append(msg)
                        logger.info(f"Action message arrived: {msg}")
                    elif msg.get('type') == 'observation':
                        arrived_observations.append(msg)
                        logger.info(f"Observation message arrived: {msg}")
        
        # Step 3: Send NEW action through network (will arrive in future)
        if self.network and action is not None:
            action_msg = {
                'type': 'action',
                'data': action.tolist() if isinstance(action, np.ndarray) else action,
                'step': self.episode_step,
                'send_time': new_time
            }
            if hasattr(self.network, 'register_packet'):
                self.network.register_packet(action_msg, flow_id=0)  # Actions on flow 0
        
        # Step 4: Execute handler with most recent arrived action (or let handler use its buffered action)
        delayed_action = None
        if arrived_actions:
            # Use the most recent action that arrived
            delayed_action = arrived_actions[-1]['data']
            if isinstance(delayed_action, list):
                delayed_action = np.array(delayed_action)
            # Debug logging
            if self.episode_step % 100 == 0:
                logger.info(f"FogSim Step {self.episode_step}: Delivering delayed action {delayed_action} from {len(arrived_actions)} arrived")
        elif self.episode_step % 100 == 0:
            logger.info(f"FogSim Step {self.episode_step}: No actions arrived yet")
        
        # Handler will use delayed_action if available, otherwise use its internal buffer
        obs, reward, success, termination, timeout, handler_info = self.handler.step_with_action(delayed_action)
        
        # Step 5: Send observation through network (it will be delayed)
        if self.network and not termination and not timeout:
            obs_msg = {
                'type': 'observation',
                'data': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                'step': self.episode_step,
                'send_time': new_time
            }
            if hasattr(self.network, 'register_packet'):
                self.network.register_packet(obs_msg, flow_id=1)  # Observations on flow 1
        
        # Step 6: Get delayed observation to return (or let caller handle None)
        delayed_obs = None
        if arrived_observations:
            # Use the most recent observation that arrived
            delayed_obs = arrived_observations[-1]['data']
            if isinstance(delayed_obs, list):
                delayed_obs = np.array(delayed_obs)
        
        # If we have a delayed observation, use it; otherwise return the fresh one
        # The handler/caller should handle the None case appropriately
        return_obs = delayed_obs if delayed_obs is not None else obs
        
        # Prepare network info
        network_info = {
            'network_delay_active': self.network is not None,
            'actions_received': len(arrived_actions),
            'observations_received': len(arrived_observations),
            'action_was_delayed': delayed_action is None and action is not None,
            'obs_was_delayed': delayed_obs is None,  # True if no delayed obs received (using fresh)
            'using_delayed_obs': delayed_obs is not None,  # True if using a delayed observation
            'using_delayed_action': delayed_action is not None,  # True if using a delayed action
            'simulation_time': new_time
        }
        
        # Add latency info if messages arrived
        if arrived_actions or arrived_observations:
            latencies = []
            for msg in arrived_actions + arrived_observations:
                if 'latency' in msg:
                    latencies.append(msg['latency'])
                elif 'send_time' in msg:
                    latencies.append((new_time - msg['send_time']) * 1000)  # Convert to ms
            if latencies:
                network_info['avg_latency_ms'] = np.mean(latencies)
        
        # Combine info
        info = {**handler_info, **network_info}
        
        self.current_obs = return_obs
        self.episode_step += 1
        
        return return_obs, reward, success, termination, timeout, info
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset simulation.
        
        Returns:
            initial_observation, info
        """
        # Reset clock
        self.clock.reset()
        
        # Reset network
        if self.network and hasattr(self.network, 'reset'):
            self.network.reset()
        
        # Reset handler
        obs, info = self.handler.reset()
        
        self.current_obs = obs
        self.episode_step = 0
        
        info['simulation_time'] = self.clock.now()
        
        logger.info("FogSim reset completed")
        return obs, info
    
    def render(self, mode: str = 'human'):
        """Render current state.""" 
        return self.handler.render(mode)
    
    def close(self):
        """Clean up resources."""
        if self.network:
            self.network.close()
        self.handler.close()
        logger.info("FogSim closed")
    
    @property
    def action_space(self):
        """Get action space from handler."""
        return self.handler.action_space
    
    @property
    def observation_space(self):
        """Get observation space from handler."""
        return self.handler.observation_space