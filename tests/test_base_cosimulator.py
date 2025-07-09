import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
import logging
from typing import Any, List

from fogsim.base import (
    BaseCoSimulator, 
    TimeManager, 
    MessageHandler,
    NetworkSimulatorProtocol
)


class MockNetworkSimulator:
    """Mock implementation of NetworkSimulatorProtocol."""
    
    def __init__(self):
        self.messages = []
        self.run_until_called = []
        self.registered_packets = []
        self.is_closed = False
    
    def run_until(self, time: float) -> None:
        self.run_until_called.append(time)
    
    def get_ready_messages(self) -> List[Any]:
        return self.messages
    
    def register_packet(self, message: Any, flow_id: int, size: float) -> str:
        packet_id = f"packet_{len(self.registered_packets)}"
        self.registered_packets.append({
            'id': packet_id,
            'message': message,
            'flow_id': flow_id,
            'size': size
        })
        return packet_id
    
    def close(self) -> None:
        self.is_closed = True


class MockRoboticsSimulator:
    """Mock implementation of robotics simulator."""
    
    def __init__(self):
        self.is_closed = False
    
    def close(self) -> None:
        self.is_closed = True


class ConcreteCoSimulator(BaseCoSimulator):
    """Concrete implementation for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handled_messages = []
        self.step_count = 0
        self.reset_count = 0
        self.render_count = 0
    
    def step(self, action: np.ndarray):
        self.step_count += 1
        return np.zeros(4), 1.0, False, {'step': self.step_count}
    
    def reset(self) -> np.ndarray:
        self.reset_count += 1
        self._time_manager.reset_time()
        return np.zeros(4)
    
    def render(self, mode: str = 'human') -> None:
        self.render_count += 1
    
    def _handle_message(self, message: Any) -> None:
        self.handled_messages.append(message)


class TestTimeManager(unittest.TestCase):
    """Test cases for TimeManager class."""
    
    def setUp(self):
        self.timestep = 0.1
        self.time_manager = TimeManager(self.timestep)
    
    def test_initialization(self):
        """Test TimeManager initialization."""
        self.assertEqual(self.time_manager.timestep, self.timestep)
        self.assertEqual(self.time_manager.current_time, self.timestep)
        self.assertIsNone(self.time_manager._last_step_time)
    
    def test_advance_time(self):
        """Test time advancement."""
        initial_time = self.time_manager.get_current_time()
        self.time_manager.advance_time()
        self.assertEqual(self.time_manager.get_current_time(), initial_time + self.timestep)
    
    def test_multiple_advances(self):
        """Test multiple time advances."""
        advances = 5
        initial_time = self.time_manager.get_current_time()
        
        for _ in range(advances):
            self.time_manager.advance_time()
        
        expected_time = initial_time + (advances * self.timestep)
        self.assertAlmostEqual(self.time_manager.get_current_time(), expected_time)
    
    def test_reset_time(self):
        """Test time reset functionality."""
        self.time_manager.advance_time()
        self.time_manager.advance_time()
        self.time_manager.reset_time()
        
        self.assertEqual(self.time_manager.get_current_time(), self.timestep)
        self.assertIsNone(self.time_manager._last_step_time)
    
    def test_get_timestep(self):
        """Test timestep getter."""
        self.assertEqual(self.time_manager.get_timestep(), self.timestep)


class TestMessageHandler(unittest.TestCase):
    """Test cases for MessageHandler class."""
    
    def setUp(self):
        self.network_sim = MockNetworkSimulator()
        self.message_handler = MessageHandler(self.network_sim)
    
    def test_flow_constants(self):
        """Test flow ID constants."""
        self.assertEqual(MessageHandler.CLIENT_TO_SERVER_FLOW, 0)
        self.assertEqual(MessageHandler.SERVER_TO_CLIENT_FLOW, 1)
    
    def test_process_messages(self):
        """Test message processing."""
        test_time = 1.5
        test_messages = ['msg1', 'msg2', 'msg3']
        self.network_sim.messages = test_messages
        
        result = self.message_handler.process_messages(test_time)
        
        self.assertEqual(result, test_messages)
        self.assertIn(test_time, self.network_sim.run_until_called)
    
    def test_send_message_default_params(self):
        """Test sending message with default parameters."""
        test_message = {'data': 'test'}
        
        msg_id = self.message_handler.send_message(test_message)
        
        self.assertEqual(msg_id, 'packet_0')
        self.assertEqual(len(self.network_sim.registered_packets), 1)
        
        packet = self.network_sim.registered_packets[0]
        self.assertEqual(packet['message'], test_message)
        self.assertEqual(packet['flow_id'], 0)
        self.assertEqual(packet['size'], 1000.0)
    
    def test_send_message_custom_params(self):
        """Test sending message with custom parameters."""
        test_message = {'data': 'test'}
        flow_id = 1
        size = 2500.0
        
        msg_id = self.message_handler.send_message(test_message, flow_id, size)
        
        packet = self.network_sim.registered_packets[0]
        self.assertEqual(packet['flow_id'], flow_id)
        self.assertEqual(packet['size'], size)
    
    @patch('fogsim.base.logger')
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        self.message_handler.process_messages(1.0)
        self.message_handler.send_message('test')
        
        # Verify logging calls were made
        self.assertTrue(mock_logger.info.called)


class TestBaseCoSimulator(unittest.TestCase):
    """Test cases for BaseCoSimulator class."""
    
    def setUp(self):
        self.network_sim = MockNetworkSimulator()
        self.robotics_sim = MockRoboticsSimulator()
        self.timestep = 0.1
        self.co_sim = ConcreteCoSimulator(
            self.network_sim, 
            self.robotics_sim, 
            self.timestep
        )
    
    def test_initialization(self):
        """Test BaseCoSimulator initialization."""
        self.assertEqual(self.co_sim.network_simulator, self.network_sim)
        self.assertEqual(self.co_sim.robotics_simulator, self.robotics_sim)
        self.assertEqual(self.co_sim.timestep, self.timestep)
        self.assertIsNotNone(self.co_sim._time_manager)
        self.assertIsNotNone(self.co_sim._message_handler)
    
    def test_internal_components(self):
        """Test that internal components are initialized properly."""
        self.assertIsInstance(self.co_sim._time_manager, TimeManager)
        self.assertIsInstance(self.co_sim._message_handler, MessageHandler)
        self.assertEqual(self.co_sim._time_manager.timestep, self.timestep)
    
    def test_flow_constants(self):
        """Test that flow constants are accessible."""
        self.assertEqual(self.co_sim.CLIENT_TO_SERVER_FLOW, 0)
        self.assertEqual(self.co_sim.SERVER_TO_CLIENT_FLOW, 1)
    
    def test_process_network_messages(self):
        """Test network message processing."""
        test_messages = ['msg1', 'msg2']
        self.network_sim.messages = test_messages
        
        result = self.co_sim._process_network_messages()
        
        self.assertEqual(result, test_messages)
        self.assertTrue(len(self.network_sim.run_until_called) > 0)
    
    def test_send_message(self):
        """Test message sending."""
        test_message = {'action': 'move'}
        
        msg_id = self.co_sim._send_message(test_message, flow_id=1, size=500.0)
        
        self.assertEqual(msg_id, 'packet_0')
        packet = self.network_sim.registered_packets[0]
        self.assertEqual(packet['message'], test_message)
        self.assertEqual(packet['flow_id'], 1)
        self.assertEqual(packet['size'], 500.0)
    
    def test_advance_time(self):
        """Test time advancement."""
        initial_time = self.co_sim.get_current_time()
        self.co_sim._advance_time()
        
        self.assertEqual(
            self.co_sim.get_current_time(), 
            initial_time + self.timestep
        )
    
    def test_time_getters(self):
        """Test time-related getters."""
        self.assertEqual(self.co_sim.get_timestep(), self.timestep)
        self.assertEqual(self.co_sim.get_current_time(), self.timestep)
        self.assertEqual(self.co_sim.current_time, self.timestep)  # Property test
    
    def test_abstract_methods_implementation(self):
        """Test that abstract methods are properly implemented."""
        # Test step
        action = np.array([1.0, 0.0])
        obs, reward, done, info = self.co_sim.step(action)
        self.assertEqual(self.co_sim.step_count, 1)
        self.assertEqual(info['step'], 1)
        
        # Test reset
        obs = self.co_sim.reset()
        self.assertEqual(self.co_sim.reset_count, 1)
        self.assertEqual(obs.shape, (4,))
        
        # Test render
        self.co_sim.render()
        self.assertEqual(self.co_sim.render_count, 1)
        
        # Test handle_message
        test_message = {'type': 'test'}
        self.co_sim._handle_message(test_message)
        self.assertIn(test_message, self.co_sim.handled_messages)
    
    def test_close(self):
        """Test resource cleanup."""
        self.assertFalse(self.network_sim.is_closed)
        self.assertFalse(self.robotics_sim.is_closed)
        
        self.co_sim.close()
        
        self.assertTrue(self.network_sim.is_closed)
        self.assertTrue(self.robotics_sim.is_closed)
    
    def test_full_simulation_cycle(self):
        """Test a complete simulation cycle."""
        # Reset
        initial_obs = self.co_sim.reset()
        self.assertEqual(self.co_sim.get_current_time(), self.timestep)
        
        # Multiple steps
        for i in range(3):
            action = np.array([1.0, 0.0])
            
            # Send action message
            self.co_sim._send_message({'action': action}, flow_id=0)
            
            # Process messages
            messages = self.co_sim._process_network_messages()
            
            # Step simulation
            obs, reward, done, info = self.co_sim.step(action)
            
            # Advance time
            self.co_sim._advance_time()
        
        # Verify state
        self.assertEqual(self.co_sim.step_count, 3)
        self.assertEqual(len(self.network_sim.registered_packets), 3)
        expected_time = self.timestep * 4  # Initial + 3 advances
        self.assertAlmostEqual(self.co_sim.get_current_time(), expected_time)
        
        # Clean up
        self.co_sim.close()


class TestIntegration(unittest.TestCase):
    """Integration tests for the co-simulator system."""
    
    def test_multiple_simulators_sharing_network(self):
        """Test multiple co-simulators sharing the same network simulator."""
        network_sim = MockNetworkSimulator()
        robotics_sim1 = MockRoboticsSimulator()
        robotics_sim2 = MockRoboticsSimulator()
        
        co_sim1 = ConcreteCoSimulator(network_sim, robotics_sim1, 0.1)
        co_sim2 = ConcreteCoSimulator(network_sim, robotics_sim2, 0.1)
        
        # Both simulators send messages
        co_sim1._send_message({'from': 'sim1'})
        co_sim2._send_message({'from': 'sim2'})
        
        # Verify both messages are in the network
        self.assertEqual(len(network_sim.registered_packets), 2)
        
        # Clean up
        co_sim1.close()
        co_sim2.close()
    
    def test_error_handling(self):
        """Test error handling in co-simulator."""
        network_sim = Mock()
        network_sim.register_packet.side_effect = Exception("Network error")
        network_sim.run_until = Mock()
        network_sim.get_ready_messages = Mock(return_value=[])
        network_sim.close = Mock()
        
        robotics_sim = MockRoboticsSimulator()
        co_sim = ConcreteCoSimulator(network_sim, robotics_sim, 0.1)
        
        # Should raise exception when trying to send message
        with self.assertRaises(Exception):
            co_sim._send_message({'test': 'data'})


if __name__ == '__main__':
    unittest.main()