"""Unit tests for the main Env class."""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from fogsim.env import Env
from fogsim.network.config import NetworkConfig
from tests.fixtures.mock_handlers import MockHandler, FailingMockHandler


class TestEnv(unittest.TestCase):
    """Test cases for the main Env class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.handler = MockHandler(observation_shape=(4,), action_shape=(2,))
        self.network_config = NetworkConfig(source_rate=1000.0, flow_weights=[1, 1])
    
    def test_initialization_with_network(self):
        """Test environment initialization with network simulation enabled."""
        env = Env(self.handler, self.network_config, enable_network=True)
        
        # Check that handler was launched
        self.assertEqual(self.handler.launch_calls, 1)
        self.assertTrue(self.handler.is_launched)
        
        # Check network simulator was created
        self.assertIsNotNone(env.network_sim)
        self.assertTrue(env.enable_network)
        
        # Clean up
        env.close()
    
    def test_initialization_without_network(self):
        """Test environment initialization without network simulation."""
        env = Env(self.handler, enable_network=False)
        
        # Check that handler was launched
        self.assertEqual(self.handler.launch_calls, 1)
        
        # Check network simulator was not created
        self.assertIsNone(env.network_sim)
        self.assertFalse(env.enable_network)
        
        # Clean up
        env.close()
    
    def test_reset(self):
        """Test environment reset functionality."""
        env = Env(self.handler, enable_network=False)
        
        # Reset environment
        observation, extra_info = env.reset()
        
        # Check that handler set_states was called (for reset)
        self.assertGreater(self.handler.set_states_calls, 0)
        
        # Check that get_states was called
        self.assertGreater(self.handler.get_states_calls, 0)
        
        # Check observation shape
        self.assertEqual(observation.shape, (4,))
        
        # Check extra info contains expected keys
        self.assertIn('network_enabled', extra_info)
        self.assertIn('timestep', extra_info)
        self.assertIn('episode', extra_info)
        
        # Clean up
        env.close()
    
    def test_step_without_network(self):
        """Test stepping without network simulation."""
        env = Env(self.handler, enable_network=False)
        
        # Reset first
        env.reset()
        
        # Step with action
        action = np.array([1.0, 0.5])
        observation, reward, success, termination, timeout, extra_info = env.step(action)
        
        # Check that handler methods were called
        self.assertGreater(self.handler.set_states_calls, 1)  # Reset + step
        self.assertGreater(self.handler.step_calls, 0)
        self.assertGreater(self.handler.get_states_calls, 1)  # Reset + step
        
        # Check return values
        self.assertEqual(observation.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(success, bool)
        self.assertIsInstance(termination, bool)
        self.assertIsInstance(timeout, bool)
        self.assertIsInstance(extra_info, dict)
        
        # Check extra info
        self.assertIn('step', extra_info)
        self.assertIn('time', extra_info)
        self.assertIn('network_latencies', extra_info)
        
        # Clean up
        env.close()
    
    @patch('fogsim.env.NSPyNetworkSimulator')
    def test_step_with_network(self, mock_network_sim_class):
        """Test stepping with network simulation."""
        # Setup mock network simulator
        mock_network_sim = Mock()
        mock_network_sim.register_packet.return_value = "msg_123"
        mock_network_sim.get_ready_messages.return_value = []
        mock_network_sim_class.return_value = mock_network_sim
        
        env = Env(self.handler, self.network_config, enable_network=True)
        
        # Reset first
        env.reset()
        
        # Step with action
        action = np.array([1.0, 0.5])
        observation, reward, success, termination, timeout, extra_info = env.step(action)
        
        # Check network simulator was used
        self.assertGreater(mock_network_sim.register_packet.call_count, 0)
        mock_network_sim.run_until.assert_called()
        mock_network_sim.get_ready_messages.assert_called()
        
        # Check return values
        self.assertEqual(observation.shape, (4,))
        self.assertIsInstance(reward, float)
        
        # Clean up
        env.close()
    
    def test_render(self):
        """Test render functionality."""
        env = Env(self.handler, enable_network=False)
        
        # Render
        frame = env.render()
        
        # Check that handler render was called
        self.assertEqual(self.handler.render_calls, 1)
        
        # Check frame shape
        self.assertEqual(frame.shape, (100, 100, 3))
        
        # Clean up
        env.close()
    
    def test_close(self):
        """Test environment cleanup."""
        env = Env(self.handler, enable_network=False)
        
        # Close environment
        env.close()
        
        # Check that handler close was called
        self.assertEqual(self.handler.close_calls, 1)
    
    def test_handler_launch_failure(self):
        """Test handling of handler launch failures."""
        failing_handler = FailingMockHandler(fail_on="launch")
        
        with self.assertRaises(RuntimeError):
            Env(failing_handler, enable_network=False)
    
    def test_get_observation_with_qpos_qvel(self):
        """Test observation extraction with qpos/qvel states."""
        # Create handler that returns qpos/qvel instead of observation
        handler = MockHandler()
        
        # Mock get_states to return qpos/qvel
        original_get_states = handler.get_states
        def mock_get_states():
            states = original_get_states()
            del states['observation']  # Remove observation
            states['qpos'] = np.array([1, 2, 3])
            states['qvel'] = np.array([4, 5, 6])
            return states
        
        handler.get_states = mock_get_states
        
        env = Env(handler, enable_network=False)
        observation, _ = env.reset()
        
        # Should concatenate qpos and qvel
        expected = np.array([1, 2, 3, 4, 5, 6])
        np.testing.assert_array_equal(observation, expected)
        
        env.close()
    
    def test_timeout_functionality(self):
        """Test timeout detection."""
        env = Env(self.handler, enable_network=False)
        env.reset()
        
        # Step many times to trigger timeout
        for _ in range(1001):  # Default timeout is 1000 steps
            observation, reward, success, termination, timeout, extra_info = env.step(np.array([0, 0]))
            if timeout:
                break
        
        # Should have timed out
        self.assertTrue(timeout)
        
        env.close()
    
    def test_network_config_default(self):
        """Test that default network config is created when none provided."""
        env = Env(self.handler, enable_network=True)
        
        # Should have created default network config
        self.assertIsNotNone(env.network_config)
        self.assertEqual(env.network_config.source_rate, 4600.0)
        
        env.close()
    
    def test_message_size_estimation(self):
        """Test message size estimation."""
        env = Env(self.handler, enable_network=False)
        
        # Test numpy array
        arr = np.zeros((10, 10), dtype=np.float64)
        size = env._estimate_message_size(arr)
        self.assertEqual(size, float(arr.nbytes))
        
        # Test list
        lst = [1, 2, 3, 4, 5]
        size = env._estimate_message_size(lst)
        self.assertEqual(size, 500.0)
        
        # Test other
        size = env._estimate_message_size("test")
        self.assertEqual(size, 1000.0)
        
        env.close()


class TestEnvIntegration(unittest.TestCase):
    """Integration tests for Env class."""
    
    def test_full_episode(self):
        """Test running a full episode."""
        handler = MockHandler()
        env = Env(handler, enable_network=False)
        
        # Reset
        observation, extra_info = env.reset()
        self.assertEqual(observation.shape, (4,))
        
        # Run episode
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 20:
            action = np.random.rand(2)
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            
            total_reward += reward
            done = termination or timeout
            steps += 1
        
        # Check episode completed
        self.assertGreater(steps, 0)
        self.assertGreater(total_reward, 0)
        
        env.close()
    
    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        handler = MockHandler()
        env = Env(handler, enable_network=False)
        
        for episode in range(3):
            observation, extra_info = env.reset()
            self.assertEqual(extra_info['episode'], episode + 1)
            
            # Run a few steps
            for step in range(5):
                action = np.random.rand(2)
                observation, reward, success, termination, timeout, extra_info = env.step(action)
                self.assertEqual(extra_info['step'], step + 1)
        
        env.close()


if __name__ == '__main__':
    unittest.main()