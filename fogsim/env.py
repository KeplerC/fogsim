"""
Legacy Environment Interface - Redirects to FogSim Core

This module provides backward compatibility for the old Env interface
by redirecting to the new streamlined FogSim core.
"""

import logging
from typing import Any, Dict, Optional, Tuple
import numpy as np

from .core import FogSim, SimulationMode
from .handlers.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class Env:
    """
    Legacy environment interface that wraps the new FogSim core.
    
    This provides backward compatibility for existing code while redirecting
    to the simplified FogSim implementation.
    """
    
    def __init__(self,
                 handler: BaseHandler,
                 network_config: Optional[Any] = None,
                 enable_network: bool = True,
                 timestep: float = 0.1,
                 mode: SimulationMode = SimulationMode.VIRTUAL,
                 **kwargs):
        """Initialize legacy environment wrapper."""
        
        logger.info("Using legacy Env interface - consider migrating to FogSim")
        
        # Create FogSim instance
        self.fogsim = FogSim(handler, mode, timestep)
        
        # Store configuration for compatibility
        self.handler = handler
        self.network_config = network_config
        self.enable_network = enable_network
        self.timestep = timestep
        self.mode = mode
        
        # Legacy state tracking
        self._step_count = 0
        self._episode_count = 0
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment."""
        self._step_count = 0
        self._episode_count += 1
        return self.fogsim.reset()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, bool, Dict[str, Any]]:
        """Step environment."""
        self._step_count += 1
        return self.fogsim.step(action)
    
    def render(self, mode: str = 'human'):
        """Render environment."""
        return self.fogsim.render(mode)
    
    def close(self):
        """Close environment."""
        self.fogsim.close()
    
    # Legacy compatibility properties
    @property
    def action_space(self):
        """Get action space."""
        return self.fogsim.action_space
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.fogsim.observation_space
    
    # Legacy methods that may be called
    def configure_network(self, config):
        """Configure network (legacy method)."""
        logger.warning("configure_network is deprecated in new FogSim API")
    
    def sync_to_time(self, time: float) -> None:
        """Legacy time sync method."""
        pass  # Handled internally by FogSim
    
    @property 
    def time_manager(self):
        """Legacy time manager access."""
        return self.fogsim.clock