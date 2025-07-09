"""Unit tests for BaseHandler abstract class."""

import unittest
from abc import ABC
import numpy as np

from fogsim.handlers.base_handler import BaseHandler


class ConcreteHandler(BaseHandler):
    """Concrete implementation of BaseHandler for testing."""
    
    def __init__(self):
        self.launched = False
        self.states = {}
        self.step_count = 0
    
    def launch(self) -> None:
        self.launched = True
    
    def set_states(self, states=None, action=None) -> None:
        if states is not None:
            self.states.update(states)
        if action is not None:
            self.states['action'] = action
    
    def get_states(self) -> dict:
        return self.states.copy()
    
    def step(self) -> None:
        self.step_count += 1
    
    def render(self) -> np.ndarray:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self) -> None:
        self.launched = False
    
    def get_extra(self) -> dict:
        return {'step_count': self.step_count, 'launched': self.launched}


class TestBaseHandler(unittest.TestCase):
    """Test cases for BaseHandler abstract class."""
    
    def test_abstract_class(self):
        """Test that BaseHandler cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseHandler()
    
    def test_concrete_implementation(self):
        """Test that concrete implementation works correctly."""
        handler = ConcreteHandler()
        
        # Test launch
        self.assertFalse(handler.is_launched)
        handler.launch()
        self.assertTrue(handler.launched)
        self.assertTrue(handler.is_launched)
        
        # Test set_states with states
        test_states = {'position': [1, 2, 3], 'velocity': [0.1, 0.2, 0.3]}
        handler.set_states(states=test_states)
        states = handler.get_states()
        self.assertEqual(states['position'], [1, 2, 3])
        self.assertEqual(states['velocity'], [0.1, 0.2, 0.3])
        
        # Test set_states with action
        test_action = np.array([1.0, 0.5])
        handler.set_states(action=test_action)
        states = handler.get_states()
        np.testing.assert_array_equal(states['action'], test_action)
        
        # Test step
        initial_count = handler.step_count
        handler.step()
        self.assertEqual(handler.step_count, initial_count + 1)
        
        # Test render
        frame = handler.render()
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(frame.dtype, np.uint8)
        
        # Test get_extra
        extra = handler.get_extra()
        self.assertIn('step_count', extra)
        self.assertIn('launched', extra)
        
        # Test close
        handler.close()
        self.assertFalse(handler.launched)
    
    def test_default_reset(self):
        """Test default reset implementation."""
        handler = ConcreteHandler()
        handler.launch()
        
        # Set some states
        handler.set_states(states={'test': 'value'})
        
        # Reset should call set_states with None
        handler.reset()
        
        # The default implementation just calls set_states(None)
        # Behavior depends on concrete implementation
    
    def test_default_is_launched(self):
        """Test default is_launched property."""
        handler = ConcreteHandler()
        
        # Should return True by default for base class
        # But our concrete implementation overrides it
        self.assertTrue(hasattr(handler, 'is_launched'))


class IncompleteHandler(BaseHandler):
    """Incomplete handler missing some abstract methods."""
    
    def launch(self) -> None:
        pass
    
    def set_states(self, states=None, action=None) -> None:
        pass
    
    def get_states(self) -> dict:
        return {}
    
    def step(self) -> None:
        pass
    
    # Missing render, close, and get_extra methods


class TestAbstractMethods(unittest.TestCase):
    """Test abstract method enforcement."""
    
    def test_incomplete_implementation_fails(self):
        """Test that incomplete implementations cannot be instantiated."""
        with self.assertRaises(TypeError):
            IncompleteHandler()
    
    def test_all_abstract_methods_present(self):
        """Test that all required abstract methods are defined."""
        abstract_methods = BaseHandler.__abstractmethods__
        expected_methods = {
            'launch', 'set_states', 'get_states', 'step', 
            'render', 'close', 'get_extra'
        }
        
        self.assertEqual(set(abstract_methods), expected_methods)


if __name__ == '__main__':
    unittest.main()