"""Integration tests for Gym handler with FogSim environment."""

import unittest
from unittest.mock import Mock, patch
import numpy as np

from fogsim import Env, GymHandler, NetworkConfig


class MockGymEnv:
    """Mock Gym environment for testing."""
    
    def __init__(self):
        self.action_space = Mock()
        self.action_space.sample.return_value = np.array([0.5, -0.5])
        self.observation_space = Mock()
        self.spec = Mock()
        self.spec.__str__ = lambda: "MockEnv-v0"
        
        self.reset_count = 0
        self.step_count = 0
        self.render_count = 0
        self.close_count = 0
        
        self._observation = np.array([1.0, 2.0, 3.0, 4.0])
        self._reward = 1.0
        self._done = False
        self._info = {}
    
    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        self._observation = np.random.rand(4)
        return self._observation
    
    def step(self, action):
        self.step_count += 1
        self._observation = np.random.rand(4)
        self._reward = np.random.rand()
        self._done = self.step_count >= 10
        self._info = {'step': self.step_count}
        return self._observation, self._reward, self._done, self._info
    
    def render(self):
        self.render_count += 1
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        self.close_count += 1


class TestGymIntegration(unittest.TestCase):
    """Integration tests for Gym handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_env = MockGymEnv()
        
        # Patch gym.make to return our mock environment
        self.gym_make_patcher = patch('fogsim.handlers.gym_handler.gym.make')
        self.mock_gym_make = self.gym_make_patcher.start()
        self.mock_gym_make.return_value = self.mock_env
        
        # Patch gym import check
        self.gym_import_patcher = patch('fogsim.handlers.gym_handler.gym', spec=True)
        self.gym_import_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.gym_make_patcher.stop()
        self.gym_import_patcher.stop()
    
    def test_gym_handler_basic_functionality(self):
        """Test basic Gym handler functionality."""
        # Create handler
        handler = GymHandler(env_name="MockEnv-v0")
        
        # Test launch
        self.assertFalse(handler.is_launched)
        handler.launch()
        self.assertTrue(handler.is_launched)
        self.assertEqual(self.mock_env.reset_count, 1)
        
        # Test get_states
        states = handler.get_states()
        self.assertIn('observation', states)
        self.assertIn('reward', states)
        self.assertIn('done', states)
        self.assertIn('step_count', states)
        
        # Test set_states with action
        action = np.array([1.0, 0.0])
        handler.set_states(action=action)
        
        # Test step
        initial_step_count = self.mock_env.step_count
        handler.step()
        self.assertEqual(self.mock_env.step_count, initial_step_count + 1)
        
        # Test render
        frame = handler.render()
        self.assertEqual(frame.shape, (64, 64, 3))
        self.assertEqual(self.mock_env.render_count, 1)
        
        # Test get_extra
        extra = handler.get_extra()
        self.assertIn('action_space', extra)
        self.assertIn('observation_space', extra)
        self.assertIn('env_name', extra)
        
        # Test close
        handler.close()
        self.assertFalse(handler.is_launched)
        self.assertEqual(self.mock_env.close_count, 1)
    
    def test_gym_handler_with_fogsim_env_no_network(self):
        """Test Gym handler integrated with FogSim environment (no network)."""
        # Create handler and environment
        handler = GymHandler(env_name="MockEnv-v0")
        env = Env(handler, enable_network=False)
        
        # Test reset
        observation, extra_info = env.reset()
        self.assertEqual(observation.shape, (4,))
        self.assertIn('action_space', extra_info)
        self.assertIn('env_name', extra_info)
        
        # Test step
        action = np.array([1.0, 0.5])
        observation, reward, success, termination, timeout, extra_info = env.step(action)
        
        self.assertEqual(observation.shape, (4,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(success, bool)
        self.assertIsInstance(termination, bool)
        self.assertIsInstance(timeout, bool)
        
        # Test render
        frame = env.render()
        self.assertEqual(frame.shape, (64, 64, 3))
        
        # Clean up
        env.close()
    
    @patch('fogsim.env.NSPyNetworkSimulator')
    def test_gym_handler_with_fogsim_env_with_network(self, mock_network_sim_class):
        """Test Gym handler integrated with FogSim environment (with network)."""
        # Setup mock network simulator
        mock_network_sim = Mock()
        mock_network_sim.register_packet.return_value = "msg_123"
        mock_network_sim.get_ready_messages.return_value = []
        mock_network_sim_class.return_value = mock_network_sim
        
        # Create handler and environment with network
        handler = GymHandler(env_name="MockEnv-v0")
        network_config = NetworkConfig(source_rate=1000.0)
        env = Env(handler, network_config, enable_network=True)
        
        # Test reset
        observation, extra_info = env.reset()
        self.assertEqual(observation.shape, (4,))
        self.assertTrue(extra_info['network_enabled'])
        
        # Test step with network
        action = np.array([1.0, 0.5])
        observation, reward, success, termination, timeout, extra_info = env.step(action)
        
        # Verify network simulator was called
        self.assertGreater(mock_network_sim.register_packet.call_count, 0)
        mock_network_sim.run_until.assert_called()
        mock_network_sim.get_ready_messages.assert_called()
        
        # Clean up
        env.close()
    
    def test_gym_handler_reset_functionality(self):
        """Test Gym handler reset functionality."""
        handler = GymHandler(env_name="MockEnv-v0")
        handler.launch()
        
        # Initial state
        states1 = handler.get_states()
        episode_count1 = states1['episode_count']
        
        # Reset
        handler.set_states(states=None)  # Reset
        states2 = handler.get_states()
        episode_count2 = states2['episode_count']
        
        # Episode count should increment
        self.assertEqual(episode_count2, episode_count1 + 1)
        self.assertEqual(states2['step_count'], 0)
        
        handler.close()
    
    def test_gym_handler_action_sampling(self):
        """Test that Gym handler samples actions when none provided."""
        handler = GymHandler(env_name="MockEnv-v0")
        handler.launch()
        
        # Step without setting action first
        initial_sample_calls = self.mock_env.action_space.sample.call_count
        handler.step()
        
        # Should have sampled an action
        self.assertGreater(self.mock_env.action_space.sample.call_count, initial_sample_calls)
        
        handler.close()
    
    def test_full_episode_with_gym_handler(self):
        """Test running a full episode with Gym handler."""
        handler = GymHandler(env_name="MockEnv-v0")
        env = Env(handler, enable_network=False)
        
        # Reset
        observation, extra_info = env.reset()
        
        # Run episode until done
        total_reward = 0
        steps = 0
        done = False
        
        while not done and steps < 20:
            action = np.random.rand(2) * 2 - 1  # Random action in [-1, 1]
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            
            total_reward += reward
            done = termination or timeout
            steps += 1
            
            # Verify observation shape
            self.assertEqual(observation.shape, (4,))
            
            # Verify step count in extra info
            self.assertEqual(extra_info['step'], steps)
        
        # Episode should have completed
        self.assertGreater(steps, 0)
        self.assertTrue(done)
        
        env.close()
    
    def test_multiple_episodes_with_gym_handler(self):
        """Test running multiple episodes with Gym handler."""
        handler = GymHandler(env_name="MockEnv-v0")
        env = Env(handler, enable_network=False)
        
        for episode in range(3):
            # Reset for new episode
            observation, extra_info = env.reset()
            self.assertEqual(extra_info['episode'], episode + 1)
            
            # Run a few steps
            for step in range(5):
                action = np.random.rand(2)
                observation, reward, success, termination, timeout, extra_info = env.step(action)
                
                # Verify step count
                self.assertEqual(extra_info['step'], step + 1)
        
        env.close()


class TestGymHandlerErrorCases(unittest.TestCase):
    """Test error cases for Gym handler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Patch gym import check
        self.gym_import_patcher = patch('fogsim.handlers.gym_handler.gym', spec=True)
        self.gym_import_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.gym_import_patcher.stop()
    
    def test_missing_env_name_and_env(self):
        """Test error when neither env_name nor env is provided."""
        with self.assertRaises(ValueError):
            GymHandler()
    
    def test_both_env_name_and_env_provided(self):
        """Test error when both env_name and env are provided."""
        mock_env = Mock()
        with self.assertRaises(ValueError):
            GymHandler(env_name="test", env=mock_env)
    
    def test_operations_before_launch(self):
        """Test that operations fail before launch."""
        handler = GymHandler(env_name="test")
        
        with self.assertRaises(RuntimeError):
            handler.set_states(action=np.array([1, 0]))
        
        with self.assertRaises(RuntimeError):
            handler.get_states()
        
        with self.assertRaises(RuntimeError):
            handler.step()
        
        with self.assertRaises(RuntimeError):
            handler.render()


if __name__ == '__main__':
    unittest.main()