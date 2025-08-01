"""OpenAI Gym handler for FogSim.

This module provides a handler implementation for OpenAI Gym environments,
wrapping them to work with the FogSim framework.
"""

from typing import Any, Dict, Optional, Union
import numpy as np
import logging

from .base_handler import BaseHandler

gym = None
# Try gymnasium first (newer), then fall back to gym (older)
try:
    import gymnasium as gym
except ImportError:
    try:
        import gym
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class GymHandler(BaseHandler):
    """Handler for OpenAI Gym environments.
    
    This handler wraps OpenAI Gym environments to work with the FogSim
    framework, providing a consistent interface for simulation control.
    
    Args:
        env_name: Name of the Gym environment to create
        env: Pre-initialized Gym environment (if env_name is not provided)
        render_mode: Rendering mode ('human', 'rgb_array', None)
        **env_kwargs: Additional keyword arguments to pass to gym.make()
    """
    
    def __init__(self, 
                 env_name: Optional[str] = None,
                 env: Optional[Any] = None,
                 render_mode: Optional[str] = None,
                 **env_kwargs):
        """Initialize the Gym handler."""
        if gym is None:
            raise ImportError(
                "OpenAI Gym is not installed. Please install it with: "
                "pip install 'fogsim[gym]'"
            )
        
        if env_name is None and env is None:
            raise ValueError("Either env_name or env must be provided")
        
        if env_name is not None and env is not None:
            raise ValueError("Only one of env_name or env should be provided")
        
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.render_mode = render_mode
        self._env = env
        self._launched = False
        self._last_observation = None
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}
        self._step_count = 0
        self._episode_count = 0
    
    def launch(self) -> None:
        """Launch the Gym environment."""
        if self._launched:
            logger.warning("Gym environment already launched")
            return
        
        if self._env is None:
            logger.info(f"Creating Gym environment: {self.env_name}")
            if self.render_mode:
                self._env = gym.make(self.env_name, render_mode=self.render_mode, **self.env_kwargs)
            else:
                self._env = gym.make(self.env_name, **self.env_kwargs)
        
        # Get initial observation (handle both old and new Gym API)
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            self._last_observation, self._last_info = reset_result
        else:
            self._last_observation = reset_result
            self._last_info = {}
        self._launched = True
        self._step_count = 0
        logger.info("Gym environment launched successfully")
    
    def reset(self) -> tuple:
        """Reset the environment and return initial observation and info."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        # Reset environment
        reset_result = self._env.reset()
        if isinstance(reset_result, tuple):
            self._last_observation, self._last_info = reset_result
        else:
            self._last_observation = reset_result
            self._last_info = {}
        
        # Reset tracking variables
        self._last_reward = 0.0
        self._last_done = False
        self._step_count = 0
        self._episode_count += 1
        
        logger.debug(f"Environment reset, episode {self._episode_count}")
        return self._last_observation, self._last_info
    
    def step_with_action(self, action) -> tuple:
        """Step environment with given action and return gym interface."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        # Step the environment (handle both old and new Gym API)
        step_result = self._env.step(action)
        if len(step_result) == 5:
            # New Gym API: (observation, reward, terminated, truncated, info)
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
            # Return in new format: obs, reward, success, termination, timeout, info
            return obs, reward, not done, terminated, truncated, info
        else:
            # Old Gym API: (observation, reward, done, info)
            obs, reward, done, info = step_result
            # Return in new format: obs, reward, success, termination, timeout, info
            return obs, reward, not done, done, False, info
    
    def set_states(self, states: Optional[Dict[str, Any]] = None, 
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states.
        
        For Gym environments, we primarily handle actions since most Gym
        environments don't support direct state setting.
        
        Args:
            states: Optional state dictionary (may trigger reset if None)
            action: Action to apply in the next step
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if states is None:
            # Reset the environment (handle both old and new Gym API)
            reset_result = self._env.reset()
            if isinstance(reset_result, tuple):
                self._last_observation, self._last_info = reset_result
            else:
                self._last_observation = reset_result
                self._last_info = {}
            self._last_reward = 0.0
            self._last_done = False
            self._step_count = 0
            self._episode_count += 1
            logger.info("Environment reset")
        
        if action is not None:
            # Store action to be applied in next step
            self._pending_action = action
    
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states.
        
        Returns:
            Dictionary containing:
            - observation: Current observation from the environment
            - reward: Last step reward
            - done: Whether episode is done
            - info: Additional information from environment
            - step_count: Number of steps in current episode
            - episode_count: Number of episodes completed
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        return {
            'observation': self._last_observation,
            'reward': self._last_reward,
            'done': self._last_done,
            'info': self._last_info,
            'step_count': self._step_count,
            'episode_count': self._episode_count
        }
    
    def step(self) -> None:
        """Step the simulation forward."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        # Use pending action if available, otherwise sample random action
        if hasattr(self, '_pending_action') and self._pending_action is not None:
            action = self._pending_action
            self._pending_action = None
        else:
            action = self._env.action_space.sample()
            logger.debug("No action provided, sampling random action")
        
        # Step the environment (handle both old and new Gym API)
        step_result = self._env.step(action)
        if len(step_result) == 5:
            # New Gym API: (observation, reward, terminated, truncated, info)
            self._last_observation, self._last_reward, terminated, truncated, self._last_info = step_result
            self._last_done = terminated or truncated
        else:
            # Old Gym API: (observation, reward, done, info)
            self._last_observation, self._last_reward, self._last_done, self._last_info = step_result
        self._step_count += 1
        
        # Handle episode termination
        if self._last_done:
            logger.info(f"Episode {self._episode_count} finished after {self._step_count} steps")
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state.
        
        Returns:
            Rendered frame as numpy array if render_mode is 'rgb_array',
            None otherwise.
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.render_mode == 'rgb_array':
            return self._env.render()
        elif self.render_mode == 'human':
            self._env.render()
            return None
        else:
            return None
    
    def close(self) -> None:
        """Close the Gym environment."""
        if self._env is not None:
            self._env.close()
            self._launched = False
            logger.info("Gym environment closed")
    
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata.
        
        Returns:
            Dictionary containing:
            - action_space: Action space of the environment
            - observation_space: Observation space of the environment
            - env_name: Name of the environment
            - render_mode: Current render mode
        """
        if not self._launched:
            return {
                'env_name': self.env_name,
                'render_mode': self.render_mode,
                'launched': False
            }
        
        return {
            'action_space': str(self._env.action_space),
            'observation_space': str(self._env.observation_space),
            'env_name': self.env_name,
            'render_mode': self.render_mode,
            'launched': True,
            'spec': str(self._env.spec) if hasattr(self._env, 'spec') else None
        }
    
    @property
    def is_launched(self) -> bool:
        """Check if the handler has been launched."""
        return self._launched
    
    @property
    def env(self) -> Any:
        """Get the underlying Gym environment."""
        return self._env
    
    @property
    def action_space(self):
        """Get action space from environment."""
        if not self._launched or self._env is None:
            raise RuntimeError("Handler not launched. Call launch() first.")
        return self._env.action_space
    
    @property
    def observation_space(self):
        """Get observation space from environment."""
        if not self._launched or self._env is None:
            raise RuntimeError("Handler not launched. Call launch() first.")
        return self._env.observation_space