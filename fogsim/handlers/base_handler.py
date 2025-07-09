"""Base handler interface for FogSim.

This module defines the abstract base class for all simulator handlers,
following the Mujoco/Roboverse interface pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseHandler(ABC):
    """Abstract base class for simulator handlers.
    
    This class defines the interface that all simulator handlers must implement
    to work with the FogSim environment. The interface follows the standard
    Mujoco/Roboverse pattern.
    """
    
    @abstractmethod
    def launch(self) -> None:
        """Launch the simulator.
        
        This method should initialize and start the simulator process or
        connection. It's called once during environment initialization.
        """
        pass
    
    @abstractmethod
    def set_states(self, states: Optional[Dict[str, Any]] = None, 
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states.
        
        Args:
            states: Optional dictionary of states to set in the simulator.
                   This could include joint positions, velocities, object poses, etc.
            action: Optional action array to apply to the simulator.
                   If provided, this represents the control input.
        """
        pass
    
    @abstractmethod
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states.
        
        Returns:
            Dictionary containing current state information such as:
            - 'observation': Current observation array
            - 'qpos': Joint positions (for Mujoco-style simulators)
            - 'qvel': Joint velocities (for Mujoco-style simulators)
            - 'objects': Object states (positions, orientations)
            - Any other simulator-specific state information
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """Step the simulation forward by one timestep.
        
        This method advances the simulation by executing one physics step.
        """
        pass
    
    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """Render the current state of the simulation.
        
        Returns:
            Optional rendered frame as a numpy array (H, W, C) in RGB format.
            Returns None if rendering is not available or not requested.
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Clean up resources and close the simulator.
        
        This method should properly shut down the simulator and release
        any resources (network connections, processes, GPU memory, etc.).
        """
        pass
    
    @abstractmethod
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata from the simulator.
        
        Returns:
            Dictionary containing additional information that doesn't fit
            into the standard state categories. This could include:
            - Debug information
            - Performance metrics
            - Simulator-specific metadata
            - Network statistics (for networked simulators)
        """
        pass
    
    def reset(self) -> None:
        """Reset the simulator to initial state.
        
        Default implementation calls set_states with None.
        Subclasses can override for custom reset behavior.
        """
        self.set_states(None)
    
    @property
    def is_launched(self) -> bool:
        """Check if the simulator has been launched.
        
        Returns:
            True if the simulator is running, False otherwise.
            
        Default implementation returns True. Subclasses should override
        if they need to track launch state.
        """
        return True