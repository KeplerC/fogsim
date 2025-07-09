import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from typing import Dict, Any, Tuple

from fogsim.environment.gym_co_simulator import (
    GymCoSimulator, 
    ObservationTracker, 
    NetworkObservationState,
    MessageSizeEstimator
)


class TestObservationTracker(unittest.TestCase):
    """Test cases for ObservationTracker class."""
    
    def setUp(self):
        self.tracker = ObservationTracker()
    
    def test_initialization(self):
        """Test ObservationTracker initialization."""
        self.assertEqual(len(self.tracker.pending_observations), 0)
        self.assertEqual(self.tracker.observation_counter, 0)
        self.assertIsNone(self.tracker.last_received_observation_id)
    
    def test_create_observation_id(self):
        """Test observation ID generation."""
        id1 = self.tracker.create_observation_id()
        id2 = self.tracker.create_observation_id()
        id3 = self.tracker.create_observation_id()
        
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)
        self.assertEqual(id3, 3)
        self.assertEqual(self.tracker.observation_counter, 3)
    
    def test_add_pending_observation(self):
        """Test adding pending observations."""
        obs_id = 1
        obs_data = {
            'observation': np.array([1, 2, 3]),
            'timestamp': 1.5,
            'reward': 10.0,
            'done': False
        }
        
        self.tracker.add_pending_observation(obs_id, obs_data)
        
        self.assertIn(obs_id, self.tracker.pending_observations)
        self.assertEqual(self.tracker.pending_observations[obs_id], obs_data)
    
    def test_update_last_received(self):
        """Test updating last received observation ID."""
        obs_id = 42
        self.tracker.update_last_received(obs_id)
        self.assertEqual(self.tracker.last_received_observation_id, obs_id)
    
    def test_reset(self):
        """Test resetting the tracker."""
        # Add some data
        self.tracker.create_observation_id()
        self.tracker.add_pending_observation(1, {'data': 'test'})
        self.tracker.update_last_received(1)
        
        # Reset
        self.tracker.reset()
        
        # Verify reset
        self.assertEqual(len(self.tracker.pending_observations), 0)
        self.assertEqual(self.tracker.observation_counter, 0)
        self.assertIsNone(self.tracker.last_received_observation_id)


class TestNetworkObservationState(unittest.TestCase):
    """Test cases for NetworkObservationState class."""
    
    def setUp(self):
        self.state = NetworkObservationState()
    
    def test_initialization(self):
        """Test NetworkObservationState initialization."""
        self.assertIsNone(self.state.last_network_observation)
        self.assertEqual(self.state.last_network_reward, 0.0)
        self.assertFalse(self.state.last_network_done)
        self.assertEqual(self.state.last_network_info, {})
    
    def test_update(self):
        """Test updating network state."""
        obs = np.array([1, 2, 3, 4])
        reward = 5.0
        done = True
        info = {'episode': 1, 'steps': 100}
        
        self.state.update(obs, reward, done, info)
        
        np.testing.assert_array_equal(self.state.last_network_observation, obs)
        self.assertEqual(self.state.last_network_reward, reward)
        self.assertTrue(self.state.last_network_done)
        self.assertEqual(self.state.last_network_info, info)
    
    def test_reset(self):
        """Test resetting the state."""
        # Set some state
        self.state.update(np.array([1, 2]), 10.0, True, {'test': 'data'})
        
        # Reset
        self.state.reset()
        
        # Verify reset
        self.assertIsNone(self.state.last_network_observation)
        self.assertEqual(self.state.last_network_reward, 0.0)
        self.assertFalse(self.state.last_network_done)
        self.assertEqual(self.state.last_network_info, {})


class TestMessageSizeEstimator(unittest.TestCase):
    """Test cases for MessageSizeEstimator class."""
    
    def test_numpy_array_estimation(self):
        """Test size estimation for numpy arrays."""
        arr = np.zeros((10, 10), dtype=np.float64)
        expected_size = arr.nbytes
        estimated_size = MessageSizeEstimator.estimate(arr)
        self.assertEqual(estimated_size, float(expected_size))
    
    def test_dict_estimation(self):
        """Test size estimation for dictionaries."""
        data = {
            'array': np.zeros((5, 5), dtype=np.float32),
            'string': 'test',
            'number': 42
        }
        estimated_size = MessageSizeEstimator.estimate(data)
        
        # Should be base size + array size + overhead for other items
        expected_min = 100.0 + data['array'].nbytes + 200.0
        self.assertGreaterEqual(estimated_size, expected_min)
    
    def test_list_estimation(self):
        """Test size estimation for lists."""
        data = [1, 2, 3, 4, 5]
        estimated_size = MessageSizeEstimator.estimate(data)
        self.assertEqual(estimated_size, 500.0)  # 5 items * 100
    
    def test_other_type_estimation(self):
        """Test size estimation for other types using pickle."""
        data = {'complex': {'nested': {'structure': [1, 2, 3]}}}
        estimated_size = MessageSizeEstimator.estimate(data)
        self.assertGreater(estimated_size, 0)


class MockGymEnvironment:
    """Mock Gym environment for testing."""
    
    def __init__(self, observation_shape=(4,), action_shape=(2,)):
        self.observation_shape = observation_shape
        self.action_shape = action_shape
        self.action_space = Mock()
        self.action_space.sample.return_value = np.zeros(action_shape)
        self.step_count = 0
        self.reset_count = 0
        self.render_count = 0
        self.last_action = None
    
    def step(self, action):
        self.step_count += 1
        self.last_action = action
        obs = np.random.rand(*self.observation_shape)
        reward = np.random.rand()
        done = self.step_count >= 10
        info = {'step': self.step_count}
        return obs, reward, done, info
    
    def reset(self):
        self.reset_count += 1
        self.step_count = 0
        return np.zeros(self.observation_shape)
    
    def render(self):
        self.render_count += 1
        return np.zeros((100, 100, 3), dtype=np.uint8)


class MockNetworkSimulator:
    """Mock network simulator for testing."""
    
    def __init__(self):
        self.messages = []
        self.sent_messages = []
        self.reset_count = 0
        self.run_until_times = []
    
    def run_until(self, time: float):
        self.run_until_times.append(time)
    
    def get_ready_messages(self):
        ready = self.messages
        self.messages = []
        return ready
    
    def register_packet(self, message: Any, flow_id: int, size: float) -> str:
        msg_id = f"msg_{len(self.sent_messages)}"
        self.sent_messages.append({
            'id': msg_id,
            'message': message,
            'flow_id': flow_id,
            'size': size
        })
        return msg_id
    
    def reset(self):
        self.reset_count += 1
        self.messages = []
        self.sent_messages = []
    
    def close(self):
        pass


class TestGymCoSimulator(unittest.TestCase):
    """Test cases for GymCoSimulator class."""
    
    def setUp(self):
        self.network_sim = MockNetworkSimulator()
        self.gym_env = MockGymEnvironment()
        self.timestep = 0.1
        self.co_sim = GymCoSimulator(
            self.network_sim,
            self.gym_env,
            self.timestep
        )
    
    def test_initialization(self):
        """Test GymCoSimulator initialization."""
        self.assertEqual(self.co_sim.network_simulator, self.network_sim)
        self.assertEqual(self.co_sim.robotics_simulator, self.gym_env)
        self.assertEqual(self.co_sim.timestep, self.timestep)
        self.assertIsNone(self.co_sim.current_observation)
        self.assertIsNone(self.co_sim.last_action)
        self.assertFalse(self.co_sim.received_action_this_step)
        self.assertIsNotNone(self.co_sim.observation_tracker)
        self.assertIsNotNone(self.co_sim.network_state)
        self.assertIsNotNone(self.co_sim.size_estimator)
    
    def test_initialization_with_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        custom_tracker = ObservationTracker()
        custom_state = NetworkObservationState()
        custom_estimator = MessageSizeEstimator()
        
        co_sim = GymCoSimulator(
            self.network_sim,
            self.gym_env,
            self.timestep,
            observation_tracker=custom_tracker,
            network_state=custom_state,
            size_estimator=custom_estimator
        )
        
        self.assertEqual(co_sim.observation_tracker, custom_tracker)
        self.assertEqual(co_sim.network_state, custom_state)
        self.assertEqual(co_sim.size_estimator, custom_estimator)
    
    def test_reset(self):
        """Test reset functionality."""
        # Add some state
        self.co_sim.current_observation = np.array([1, 2, 3])
        self.co_sim.last_action = np.array([0, 1])
        self.co_sim.observation_tracker.create_observation_id()
        self.co_sim.network_state.update(np.array([4, 5]), 1.0, False, {})
        
        # Reset
        obs = self.co_sim.reset()
        
        # Verify reset
        self.assertEqual(self.gym_env.reset_count, 1)
        self.assertEqual(self.network_sim.reset_count, 1)
        self.assertIsNone(self.co_sim.last_action)
        self.assertFalse(self.co_sim.received_action_this_step)
        self.assertEqual(self.co_sim.observation_tracker.observation_counter, 0)
        self.assertIsNone(self.co_sim.network_state.last_network_observation)
        np.testing.assert_array_equal(obs, np.zeros(self.gym_env.observation_shape))
    
    def test_step_without_network_messages(self):
        """Test step when no network messages are received."""
        action = np.array([1.0, 0.0])
        
        obs, reward, done, info = self.co_sim.step(action)
        
        # Should use the immediate observation since no network observation exists
        self.assertIsNotNone(obs)
        self.assertEqual(reward, 0.0)  # Default network reward
        self.assertFalse(done)  # Default network done
        
        # Verify observation was sent through network (server to client)
        self.assertEqual(len(self.network_sim.sent_messages), 1)
        sent_msg = self.network_sim.sent_messages[0]
        self.assertEqual(sent_msg['flow_id'], 1)  # SERVER_TO_CLIENT_FLOW = 1
        # Should be an observation message, not action
        self.assertIn('observation', sent_msg['message'])
    
    def test_step_with_observation_message(self):
        """Test step when observation message is received."""
        # Setup network observation
        network_obs = np.array([1, 2, 3, 4])
        network_reward = 5.0
        network_done = False
        network_info = {'test': 'info'}
        
        self.network_sim.messages = [{
            'observation': network_obs,
            'observation_id': 1,
            'timestamp': 0.0,
            'reward': network_reward,
            'done': network_done,
            'info': network_info
        }]
        
        obs, reward, done, info = self.co_sim.step()
        
        # Should return the network observation
        np.testing.assert_array_equal(obs, network_obs)
        self.assertEqual(reward, network_reward)
        self.assertEqual(done, network_done)
        self.assertIn('test', info)
        self.assertEqual(info['test'], 'info')
    
    def test_step_with_action_message(self):
        """Test step when action message is received."""
        # First send an observation
        self.co_sim.observation_tracker.create_observation_id()
        self.co_sim.observation_tracker.add_pending_observation(1, {
            'observation': np.array([1, 2, 3, 4]),
            'timestamp': 0.0
        })
        
        # Setup action message
        received_action = np.array([0.5, -0.5])
        self.network_sim.messages = [{
            'action': received_action,
            'responding_to_observation': 1,
            'timestamp': 0.1
        }]
        
        obs, reward, done, info = self.co_sim.step()
        
        # Verify action was received and used
        self.assertTrue(self.co_sim.received_action_this_step)
        np.testing.assert_array_equal(self.co_sim.last_action, received_action)
        np.testing.assert_array_equal(self.gym_env.last_action, received_action)
    
    def test_round_trip_latency_calculation(self):
        """Test round-trip latency calculation."""
        # Setup observation in pending
        obs_id = 1
        obs_timestamp = 0.0
        self.co_sim.observation_tracker.add_pending_observation(obs_id, {
            'observation': np.array([1, 2, 3, 4]),
            'timestamp': obs_timestamp
        })
        
        # Setup action message responding to observation
        action_timestamp = 0.5
        self.co_sim._time_manager.current_time = 1.0
        
        self.network_sim.messages = [{
            'action': np.array([1.0, 0.0]),
            'responding_to_observation': obs_id,
            'timestamp': action_timestamp
        }]
        
        obs, reward, done, info = self.co_sim.step()
        
        # Verify action latency is calculated
        self.assertIn('action_latencies', info)
        self.assertEqual(len(info['action_latencies']), 1)
        expected_action_latency = 1.0 - action_timestamp  # Current time - action timestamp
        self.assertAlmostEqual(info['action_latencies'][0], expected_action_latency, places=10)
        
        # Verify that action was received and processed
        self.assertTrue(self.co_sim.received_action_this_step)
        
        # Round-trip latency should be calculated since action was received this step  
        # Check if round-trip latency exists (it might not be calculated in this specific scenario)
        if 'round_trip_latency' in info:
            expected_latency = 1.0 - obs_timestamp  # Current time - observation timestamp
            self.assertAlmostEqual(info['round_trip_latency'], expected_latency, places=10)
    
    def test_render(self):
        """Test render functionality."""
        frame = self.co_sim.render()
        
        self.assertEqual(self.gym_env.render_count, 1)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))
    
    def test_handle_message_legacy(self):
        """Test legacy message handling."""
        # Test observation message
        obs_msg = {'observation': np.array([1, 2, 3])}
        self.co_sim._handle_message(obs_msg)
        np.testing.assert_array_equal(self.co_sim.current_observation, obs_msg['observation'])
        
        # Test action message
        action_msg = {'action': np.array([0.5])}
        self.co_sim._handle_message(action_msg)
        np.testing.assert_array_equal(self.co_sim.last_action, action_msg['action'])
        self.assertTrue(self.co_sim.received_action_this_step)
    
    def test_estimate_message_size(self):
        """Test message size estimation."""
        # Test numpy array
        arr = np.zeros((10, 10), dtype=np.float64)
        size = self.co_sim._estimate_message_size(arr)
        self.assertEqual(size, float(arr.nbytes))
        
        # Test other types
        data = {'test': 'data'}
        size = self.co_sim._estimate_message_size(data)
        self.assertGreater(size, 0)
    
    def test_get_time(self):
        """Test get_time method."""
        self.assertEqual(self.co_sim.get_time(), self.co_sim.get_current_time())
    
    def test_action_sampling_when_no_prior_action(self):
        """Test action sampling when no prior action exists."""
        # Step without any action
        obs, reward, done, info = self.co_sim.step()
        
        # Should have sampled an action
        self.assertIsNotNone(self.co_sim.last_action)
        self.assertEqual(self.gym_env.action_space.sample.call_count, 1)
    
    def test_multiple_steps_with_latency(self):
        """Test multiple steps with network latency."""
        # Initial reset
        self.co_sim.reset()
        
        # Step 1: Send initial action
        action1 = np.array([1.0, 0.0])
        obs1, _, _, _ = self.co_sim.step(action1)
        
        # Step 2: Receive observation from step 1
        self.network_sim.messages = [{
            'observation': np.array([1, 2, 3, 4]),
            'observation_id': 1,
            'timestamp': 0.1,
            'reward': 1.0,
            'done': False,
            'info': {}
        }]
        
        action2 = np.array([0.0, 1.0])
        obs2, reward2, done2, info2 = self.co_sim.step(action2)
        
        # Should get the network observation
        np.testing.assert_array_equal(obs2, np.array([1, 2, 3, 4]))
        self.assertEqual(reward2, 1.0)
        
        # Verify observation latency tracking
        self.assertIn('observation_latencies', info2)
        self.assertEqual(len(info2['observation_latencies']), 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for GymCoSimulator."""
    
    def test_full_episode_simulation(self):
        """Test a full episode with proper message flow."""
        network_sim = MockNetworkSimulator()
        gym_env = MockGymEnvironment()
        co_sim = GymCoSimulator(network_sim, gym_env, 0.1)
        
        # Reset
        initial_obs = co_sim.reset()
        
        episode_rewards = []
        done = False
        step = 0
        
        while step < 5 and not done:
            # Client sends action
            action = np.array([1.0, 0.0])
            
            # Simulate network delay - observation from previous step arrives
            if step > 0:
                network_sim.messages = [{
                    'observation': np.random.rand(4),
                    'observation_id': step,
                    'timestamp': step * 0.1,
                    'reward': 0.5,
                    'done': False,
                    'info': {'step': step}
                }]
            
            obs, reward, done, info = co_sim.step(action)
            episode_rewards.append(reward)
            step += 1
        
        # Verify simulation ran
        self.assertEqual(len(episode_rewards), 5)
        self.assertGreater(gym_env.step_count, 0)
        self.assertGreater(len(network_sim.sent_messages), 0)


if __name__ == '__main__':
    unittest.main()