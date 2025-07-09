"""Mock handler implementations for testing."""

from typing import Any, Dict, Optional
import numpy as np
from fogsim.handlers.base_handler import BaseHandler


class MockHandler(BaseHandler):
    """Mock handler for testing purposes."""
    
    def __init__(self, observation_shape: tuple = (4,), action_shape: tuple = (2,)):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self._launched = False
        self._step_count = 0
        self._episode_count = 0
        self._current_observation = np.zeros(observation_shape)
        self._last_action = None
        self._last_states = None
        
        # Call tracking
        self.launch_calls = 0
        self.set_states_calls = 0
        self.get_states_calls = 0
        self.step_calls = 0
        self.render_calls = 0
        self.close_calls = 0
        self.get_extra_calls = 0
    
    def launch(self) -> None:
        """Mock launch implementation."""
        self.launch_calls += 1
        self._launched = True
        self._current_observation = np.random.rand(*self.observation_shape)
    
    def set_states(self, states: Optional[Dict[str, Any]] = None, 
                   action: Optional[np.ndarray] = None) -> None:
        """Mock set_states implementation."""
        self.set_states_calls += 1
        
        if states is None:
            # Reset
            self._step_count = 0
            self._episode_count += 1
            self._current_observation = np.random.rand(*self.observation_shape)
        else:
            self._last_states = states
        
        if action is not None:
            self._last_action = action
    
    def get_states(self) -> Dict[str, Any]:
        """Mock get_states implementation."""
        self.get_states_calls += 1
        
        return {
            'observation': self._current_observation.copy(),
            'reward': np.random.rand(),
            'done': self._step_count >= 10,
            'step_count': self._step_count,
            'episode_count': self._episode_count,
            'last_action': self._last_action.copy() if self._last_action is not None else None
        }
    
    def step(self) -> None:
        """Mock step implementation."""
        self.step_calls += 1
        self._step_count += 1
        
        # Update observation
        self._current_observation = np.random.rand(*self.observation_shape)
    
    def render(self) -> Optional[np.ndarray]:
        """Mock render implementation."""
        self.render_calls += 1
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    def close(self) -> None:
        """Mock close implementation."""
        self.close_calls += 1
        self._launched = False
    
    def get_extra(self) -> Dict[str, Any]:
        """Mock get_extra implementation."""
        self.get_extra_calls += 1
        
        return {
            'handler_type': 'mock',
            'observation_shape': self.observation_shape,
            'action_shape': self.action_shape,
            'launched': self._launched
        }
    
    @property
    def is_launched(self) -> bool:
        """Check if mock handler is launched."""
        return self._launched
    
    def reset_call_counts(self) -> None:
        """Reset all call counters."""
        self.launch_calls = 0
        self.set_states_calls = 0
        self.get_states_calls = 0
        self.step_calls = 0
        self.render_calls = 0
        self.close_calls = 0
        self.get_extra_calls = 0


class FailingMockHandler(MockHandler):
    """Mock handler that fails at specific operations."""
    
    def __init__(self, fail_on: str = "launch", **kwargs):
        super().__init__(**kwargs)
        self.fail_on = fail_on
    
    def launch(self) -> None:
        if self.fail_on == "launch":
            raise RuntimeError("Mock launch failure")
        super().launch()
    
    def step(self) -> None:
        if self.fail_on == "step":
            raise RuntimeError("Mock step failure")
        super().step()
    
    def render(self) -> Optional[np.ndarray]:
        if self.fail_on == "render":
            raise RuntimeError("Mock render failure")
        return super().render()
    
    def close(self) -> None:
        if self.fail_on == "close":
            raise RuntimeError("Mock close failure")
        super().close()


class SlowMockHandler(MockHandler):
    """Mock handler with configurable delays for performance testing."""
    
    def __init__(self, step_delay: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.step_delay = step_delay
    
    def step(self) -> None:
        import time
        if self.step_delay > 0:
            time.sleep(self.step_delay)
        super().step()