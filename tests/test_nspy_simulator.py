import unittest
from unittest.mock import Mock, MagicMock, patch
import simpy
from typing import Dict, List

from fogsim.network.nspy_simulator import NSPyNetworkSimulator, PacketTracker


class TestPacketTracker(unittest.TestCase):
    """Test cases for PacketTracker class."""
    
    def setUp(self):
        self.tracker = PacketTracker()
    
    def test_initialization(self):
        """Test PacketTracker initialization."""
        self.assertEqual(len(self.tracker.pending_packets), 0)
        self.assertEqual(len(self.tracker.ready_messages), 0)
        self.assertEqual(len(self.tracker.last_checked_arrivals), 0)
    
    def test_add_pending_packet(self):
        """Test adding a pending packet."""
        msg_id = "test-123"
        message = {"data": "test"}
        sent_time = 1.5
        flow_id = 0
        
        self.tracker.add_pending_packet(msg_id, message, sent_time, flow_id)
        
        self.assertIn(msg_id, self.tracker.pending_packets)
        self.assertEqual(self.tracker.pending_packets[msg_id], (message, sent_time, flow_id))
        self.assertIn(flow_id, self.tracker.last_checked_arrivals)
        self.assertEqual(self.tracker.last_checked_arrivals[flow_id], 0)
    
    def test_add_multiple_flows(self):
        """Test adding packets from multiple flows."""
        self.tracker.add_pending_packet("msg1", "data1", 1.0, 0)
        self.tracker.add_pending_packet("msg2", "data2", 2.0, 1)
        self.tracker.add_pending_packet("msg3", "data3", 3.0, 0)
        
        self.assertEqual(len(self.tracker.pending_packets), 3)
        self.assertEqual(len(self.tracker.last_checked_arrivals), 2)  # Two flows
        self.assertIn(0, self.tracker.last_checked_arrivals)
        self.assertIn(1, self.tracker.last_checked_arrivals)
    
    def test_mark_packet_delivered(self):
        """Test marking a packet as delivered."""
        msg_id = "test-123"
        message = {"data": "test"}
        
        self.tracker.add_pending_packet(msg_id, message, 1.0, 0)
        self.tracker.mark_packet_delivered(msg_id)
        
        self.assertNotIn(msg_id, self.tracker.pending_packets)
        self.assertIn(msg_id, self.tracker.ready_messages)
        self.assertEqual(self.tracker.ready_messages[msg_id], message)
    
    def test_mark_nonexistent_packet_delivered(self):
        """Test marking a non-existent packet as delivered."""
        # Should not raise exception
        self.tracker.mark_packet_delivered("nonexistent-id")
        self.assertEqual(len(self.tracker.ready_messages), 0)
    
    def test_get_ready_messages(self):
        """Test getting ready messages."""
        messages = [
            {"id": 1, "data": "msg1"},
            {"id": 2, "data": "msg2"},
            {"id": 3, "data": "msg3"}
        ]
        
        # Add and mark packets as delivered
        for i, msg in enumerate(messages):
            msg_id = f"msg-{i}"
            self.tracker.add_pending_packet(msg_id, msg, float(i), 0)
            self.tracker.mark_packet_delivered(msg_id)
        
        # Get ready messages
        ready = self.tracker.get_ready_messages()
        
        self.assertEqual(len(ready), 3)
        self.assertIn(messages[0], ready)
        self.assertIn(messages[1], ready)
        self.assertIn(messages[2], ready)
        
        # Verify messages are cleared after retrieval
        self.assertEqual(len(self.tracker.ready_messages), 0)
        self.assertEqual(len(self.tracker.get_ready_messages()), 0)
    
    def test_reset(self):
        """Test resetting the tracker."""
        # Add some data
        self.tracker.add_pending_packet("msg1", "data1", 1.0, 0)
        self.tracker.add_pending_packet("msg2", "data2", 2.0, 1)
        self.tracker.mark_packet_delivered("msg1")
        
        # Reset
        self.tracker.reset()
        
        # Verify all data is cleared
        self.assertEqual(len(self.tracker.pending_packets), 0)
        self.assertEqual(len(self.tracker.ready_messages), 0)
        self.assertEqual(len(self.tracker.last_checked_arrivals), 0)


class MockVirtualClockServer:
    """Mock VirtualClockServer for testing."""
    
    def __init__(self, env, source_rate, weights, debug=False):
        self.env = env
        self.source_rate = source_rate
        self.weights = weights
        self.debug = debug
        self.out = None
        self.packets = []
    
    def put(self, packet):
        self.packets.append(packet)


class MockPacketSink:
    """Mock PacketSink for testing."""
    
    def __init__(self, env):
        self.env = env
        self.arrivals: Dict[int, List[float]] = {}
    
    def add_arrival(self, flow_id: int, arrival_time: float):
        """Add a mock arrival for testing."""
        if flow_id not in self.arrivals:
            self.arrivals[flow_id] = []
        self.arrivals[flow_id].append(arrival_time)


class TestNSPyNetworkSimulator(unittest.TestCase):
    """Test cases for NSPyNetworkSimulator class."""
    
    def setUp(self):
        self.source_rate = 5000.0
        self.weights = [1, 2]
        self.simulator = NSPyNetworkSimulator(
            source_rate=self.source_rate,
            weights=self.weights,
            debug=True
        )
    
    @patch('fogsim.network.nspy_simulator.VirtualClockServer', MockVirtualClockServer)
    @patch('fogsim.network.nspy_simulator.PacketSink', MockPacketSink)
    def test_initialization(self):
        """Test NSPyNetworkSimulator initialization."""
        sim = NSPyNetworkSimulator(
            source_rate=self.source_rate,
            weights=self.weights,
            debug=True
        )
        
        self.assertIsInstance(sim.env, simpy.Environment)
        self.assertEqual(sim.source_rate, self.source_rate)
        self.assertEqual(sim.flow_weights, self.weights)
        self.assertTrue(sim.debug)
        self.assertIsNotNone(sim.packet_tracker)
        self.assertIsInstance(sim.packet_tracker, PacketTracker)
    
    def test_initialization_with_custom_dependencies(self):
        """Test initialization with custom dependencies."""
        custom_env = simpy.Environment()
        custom_tracker = PacketTracker()
        
        sim = NSPyNetworkSimulator(
            source_rate=self.source_rate,
            weights=self.weights,
            env=custom_env,
            packet_tracker=custom_tracker
        )
        
        self.assertEqual(sim.env, custom_env)
        self.assertEqual(sim.packet_tracker, custom_tracker)
    
    @patch('fogsim.network.nspy_simulator.uuid.uuid4')
    @patch('fogsim.network.nspy_simulator.VirtualClockServer', MockVirtualClockServer)
    @patch('fogsim.network.nspy_simulator.PacketSink', MockPacketSink)
    def test_register_packet(self, mock_uuid):
        """Test packet registration."""
        mock_uuid.return_value = "test-uuid-123"
        
        sim = NSPyNetworkSimulator()
        message = {"action": "move", "value": 10}
        flow_id = 1
        size = 1500.0
        
        msg_id = sim.register_packet(message, flow_id, size)
        
        self.assertEqual(msg_id, "test-uuid-123")
        self.assertIn(msg_id, sim.packet_tracker.pending_packets)
        
        # Verify packet was sent to server
        self.assertEqual(len(sim.vc_server.packets), 1)
        packet = sim.vc_server.packets[0]
        self.assertEqual(packet.flow_id, flow_id)
        self.assertEqual(packet.size, size)
        self.assertEqual(packet.packet_id, msg_id)
    
    def test_run_until_no_advance(self):
        """Test run_until when time hasn't advanced."""
        initial_time = self.simulator.env.now
        
        # Run until current time or earlier
        self.simulator.run_until(initial_time)
        
        # Should not advance
        self.assertEqual(self.simulator.env.now, initial_time)
    
    def test_run_until_advance(self):
        """Test run_until when time should advance."""
        initial_time = self.simulator.env.now
        future_time = initial_time + 10.0
        
        self.simulator.run_until(future_time)
        
        # Should advance to future time
        self.assertEqual(self.simulator.env.now, future_time)
    
    @patch('fogsim.network.nspy_simulator.VirtualClockServer', MockVirtualClockServer)
    @patch('fogsim.network.nspy_simulator.PacketSink', MockPacketSink)
    def test_process_arrivals(self):
        """Test processing packet arrivals."""
        sim = NSPyNetworkSimulator()
        
        # Register some packets
        msg1_id = sim.register_packet({"msg": 1}, flow_id=0, size=100)
        msg2_id = sim.register_packet({"msg": 2}, flow_id=0, size=100)
        msg3_id = sim.register_packet({"msg": 3}, flow_id=1, size=200)
        
        # Mock some arrivals
        sim.sink.add_arrival(0, 1.0)
        sim.sink.add_arrival(0, 2.0)
        sim.sink.add_arrival(1, 1.5)
        
        # Process arrivals up to time 2.5
        sim._process_arrivals(2.5)
        
        # All packets should be marked as delivered
        ready_messages = sim.get_ready_messages()
        self.assertEqual(len(ready_messages), 3)
        
        # Verify pending packets are cleared
        self.assertEqual(len(sim.packet_tracker.pending_packets), 0)
    
    def test_get_ready_messages(self):
        """Test getting ready messages."""
        # Add some ready messages directly
        test_messages = [{"id": 1}, {"id": 2}, {"id": 3}]
        for i, msg in enumerate(test_messages):
            self.simulator.packet_tracker.ready_messages[f"msg-{i}"] = msg
        
        # Get messages
        messages = self.simulator.get_ready_messages()
        
        self.assertEqual(len(messages), 3)
        for msg in test_messages:
            self.assertIn(msg, messages)
        
        # Verify messages are cleared
        self.assertEqual(len(self.simulator.get_ready_messages()), 0)
    
    @patch('fogsim.network.nspy_simulator.VirtualClockServer', MockVirtualClockServer)
    @patch('fogsim.network.nspy_simulator.PacketSink', MockPacketSink)
    def test_reset(self):
        """Test resetting the simulator."""
        # Add some state
        self.simulator.register_packet({"test": "data"}, flow_id=0)
        
        # Reset
        self.simulator.reset()
        
        # Verify reset
        self.assertEqual(self.simulator.env.now, 0)
        self.assertEqual(len(self.simulator.packet_tracker.pending_packets), 0)
        self.assertEqual(len(self.simulator.packet_tracker.ready_messages), 0)
        self.assertIsNotNone(self.simulator.vc_server)
        self.assertIsNotNone(self.simulator.sink)
    
    def test_close(self):
        """Test closing the simulator."""
        # Should not raise any exceptions
        self.simulator.close()
    
    @patch('fogsim.network.nspy_simulator.logger')
    def test_logging(self, mock_logger):
        """Test that appropriate logging occurs."""
        sim = NSPyNetworkSimulator()
        sim.register_packet({"test": "data"})
        sim.run_until(1.0)
        sim.get_ready_messages()
        
        # Verify logging calls were made
        self.assertTrue(mock_logger.info.called)


class TestIntegration(unittest.TestCase):
    """Integration tests for the network simulator."""
    
    @patch('fogsim.network.nspy_simulator.VirtualClockServer', MockVirtualClockServer)
    @patch('fogsim.network.nspy_simulator.PacketSink', MockPacketSink)
    def test_full_message_flow(self):
        """Test complete message flow through the simulator."""
        sim = NSPyNetworkSimulator(source_rate=10000.0, weights=[1, 1])
        
        # Send multiple messages
        messages = [
            {"type": "sensor", "value": 100},
            {"type": "control", "action": "move"},
            {"type": "status", "ok": True}
        ]
        
        msg_ids = []
        for i, msg in enumerate(messages):
            msg_id = sim.register_packet(msg, flow_id=i % 2, size=100 + i * 50)
            msg_ids.append(msg_id)
        
        # Simulate arrivals at different times
        sim.sink.add_arrival(0, 0.5)  # First message (flow 0)
        sim.sink.add_arrival(1, 0.7)  # Second message (flow 1)
        sim.sink.add_arrival(0, 1.0)  # Third message (flow 0)
        
        # Run simulation to process all arrivals
        sim.run_until(2.0)
        
        # Get ready messages
        ready = sim.get_ready_messages()
        
        # Verify all messages are delivered
        self.assertEqual(len(ready), 3)
        for msg in messages:
            self.assertIn(msg, ready)
    
    def test_multiple_flows_with_different_weights(self):
        """Test that flow weights affect packet scheduling."""
        # This is more of a placeholder test since we're mocking the actual
        # VirtualClockServer behavior. In real testing, we'd verify that
        # packets from flows with lower weights get scheduled earlier.
        sim = NSPyNetworkSimulator(source_rate=1000.0, weights=[1, 4])
        
        # Send packets on both flows
        sim.register_packet({"flow": 0}, flow_id=0, size=1000)
        sim.register_packet({"flow": 1}, flow_id=1, size=1000)
        
        # In real implementation, flow 0 should be scheduled before flow 1
        # due to lower weight


if __name__ == '__main__':
    unittest.main()