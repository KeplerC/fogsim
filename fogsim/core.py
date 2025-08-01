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
                 timestep: float = 0.1):
        """
        Initialize FogSim.
        
        Args:
            handler: Environment handler (GymHandler, CarlaHandler, etc.)
            mode: Simulation mode
            timestep: Simulation timestep in seconds
        """
        self.handler = handler
        self.mode = mode
        self.timestep = timestep
        
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
            # No network simulation in virtual mode
            return None
        elif self.mode == SimulationMode.SIMULATED_NET:
            from .network.nspy_simulator import NSPyNetworkSimulator
            return NSPyNetworkSimulator()
        else:  # REAL_NET
            from .network.real_network import RealNetworkTransport
            return RealNetworkTransport()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step.
        
        Args:
            action: Action to execute
            
        Returns:
            observation, reward, success, termination, timeout, info
        """
        # Advance time
        self.clock.advance()
        
        # Process network messages if applicable
        network_info = {}
        if self.network:
            if hasattr(self.network, 'process_messages'):
                messages = self.network.process_messages(self.clock.now())
            elif hasattr(self.network, 'get_ready_messages'):
                # For NSPy simulator, run until current time first
                if hasattr(self.network, 'run_until'):
                    self.network.run_until(self.clock.now())
                messages = self.network.get_ready_messages()
            else:
                messages = []
            
            network_info = {
                'network_latencies': [{'latency': msg.get('latency', 0)} for msg in messages],
                'num_messages_received': len(messages)
            }
        
        # Execute handler step
        obs, reward, success, termination, timeout, handler_info = self.handler.step_with_action(action)
        
        # Send network message if applicable
        if self.network and not termination and not timeout:
            message = {
                'step': self.episode_step,
                'observation': obs.tolist() if isinstance(obs, np.ndarray) else obs,
                'action': action.tolist() if isinstance(action, np.ndarray) else action
            }
            
            if hasattr(self.network, 'send_message'):
                self.network.send_message(message)
            elif hasattr(self.network, 'register_packet'):
                self.network.register_packet(message)
        
        # Combine info
        info = {**handler_info, **network_info, 'simulation_time': self.clock.now()}
        
        self.current_obs = obs
        self.episode_step += 1
        
        return obs, reward, success, termination, timeout, info
    
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