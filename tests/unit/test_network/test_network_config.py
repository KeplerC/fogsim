"""Unit tests for NetworkConfig class."""

import unittest
from fogsim.network.config import (
    NetworkConfig, TopologyConfig, FlowConfig, TokenBucketConfig,
    TopologyType, CongestionControl, SchedulerType,
    get_low_latency_config, get_satellite_config, get_iot_config
)


class TestTokenBucketConfig(unittest.TestCase):
    """Test cases for TokenBucketConfig."""
    
    def test_initialization(self):
        """Test TokenBucketConfig initialization."""
        config = TokenBucketConfig(rate=1000.0, size=500.0)
        
        self.assertEqual(config.rate, 1000.0)
        self.assertEqual(config.size, 500.0)
        self.assertIsNone(config.peak_rate)
        self.assertIsNone(config.peak_size)
    
    def test_two_rate_config(self):
        """Test two-rate token bucket configuration."""
        config = TokenBucketConfig(
            rate=1000.0, size=500.0,
            peak_rate=2000.0, peak_size=1000.0
        )
        
        self.assertEqual(config.peak_rate, 2000.0)
        self.assertEqual(config.peak_size, 1000.0)


class TestFlowConfig(unittest.TestCase):
    """Test cases for FlowConfig."""
    
    def test_default_initialization(self):
        """Test FlowConfig with default values."""
        config = FlowConfig(flow_id=1)
        
        self.assertEqual(config.flow_id, 1)
        self.assertEqual(config.weight, 1)
        self.assertEqual(config.priority, 0)
        self.assertIsNone(config.congestion_control)
        self.assertIsNone(config.token_bucket)
    
    def test_full_initialization(self):
        """Test FlowConfig with all parameters."""
        token_bucket = TokenBucketConfig(rate=1000.0, size=500.0)
        config = FlowConfig(
            flow_id=2,
            weight=5,
            priority=10,
            congestion_control=CongestionControl.BBR,
            token_bucket=token_bucket
        )
        
        self.assertEqual(config.flow_id, 2)
        self.assertEqual(config.weight, 5)
        self.assertEqual(config.priority, 10)
        self.assertEqual(config.congestion_control, CongestionControl.BBR)
        self.assertEqual(config.token_bucket, token_bucket)


class TestTopologyConfig(unittest.TestCase):
    """Test cases for TopologyConfig."""
    
    def test_default_initialization(self):
        """Test TopologyConfig with default values."""
        config = TopologyConfig()
        
        self.assertEqual(config.topology_type, TopologyType.SIMPLE)
        self.assertEqual(config.num_hosts, 4)
        self.assertEqual(config.num_switches, 1)
        self.assertEqual(config.link_bandwidth, 1e6)
        self.assertEqual(config.link_delay, 0.001)
        self.assertIsNone(config.topology_file)
        self.assertIsNone(config.custom_topology)
    
    def test_custom_initialization(self):
        """Test TopologyConfig with custom values."""
        custom_topo = {'nodes': [], 'edges': []}
        config = TopologyConfig(
            topology_type=TopologyType.FATTREE,
            num_hosts=16,
            num_switches=20,
            link_bandwidth=10e6,
            link_delay=0.005,
            topology_file="/path/to/topo.graphml",
            custom_topology=custom_topo
        )
        
        self.assertEqual(config.topology_type, TopologyType.FATTREE)
        self.assertEqual(config.num_hosts, 16)
        self.assertEqual(config.num_switches, 20)
        self.assertEqual(config.link_bandwidth, 10e6)
        self.assertEqual(config.link_delay, 0.005)
        self.assertEqual(config.topology_file, "/path/to/topo.graphml")
        self.assertEqual(config.custom_topology, custom_topo)


class TestNetworkConfig(unittest.TestCase):
    """Test cases for NetworkConfig."""
    
    def test_default_initialization(self):
        """Test NetworkConfig with default values."""
        config = NetworkConfig()
        
        # Check default values
        self.assertEqual(config.source_rate, 4600.0)
        self.assertEqual(config.flow_weights, [1, 1])
        self.assertEqual(config.scheduler, SchedulerType.VIRTUAL_CLOCK)
        self.assertEqual(config.packet_loss_rate, 0.0)
        self.assertFalse(config.enable_shaping)
        self.assertEqual(config.congestion_control, CongestionControl.CUBIC)
        self.assertFalse(config.enable_red)
        self.assertEqual(config.simulation_time, 100.0)
        self.assertFalse(config.debug)
        self.assertIsNone(config.seed)
    
    def test_custom_initialization(self):
        """Test NetworkConfig with custom values."""
        topology = TopologyConfig(topology_type=TopologyType.FATTREE)
        flows = [FlowConfig(flow_id=0), FlowConfig(flow_id=1)]
        
        config = NetworkConfig(
            topology=topology,
            scheduler=SchedulerType.WFQ,
            source_rate=10000.0,
            flow_weights=[2, 3],
            flows=flows,
            packet_loss_rate=0.01,
            enable_shaping=True,
            congestion_control=CongestionControl.BBR,
            enable_red=True,
            simulation_time=200.0,
            debug=True,
            seed=42
        )
        
        self.assertEqual(config.topology, topology)
        self.assertEqual(config.scheduler, SchedulerType.WFQ)
        self.assertEqual(config.source_rate, 10000.0)
        self.assertEqual(config.flow_weights, [2, 3])
        self.assertEqual(config.flows, flows)
        self.assertEqual(config.packet_loss_rate, 0.01)
        self.assertTrue(config.enable_shaping)
        self.assertEqual(config.congestion_control, CongestionControl.BBR)
        self.assertTrue(config.enable_red)
        self.assertEqual(config.simulation_time, 200.0)
        self.assertTrue(config.debug)
        self.assertEqual(config.seed, 42)
    
    def test_post_init_flow_weights_extension(self):
        """Test that flow_weights are extended when more flows are added."""
        flows = [FlowConfig(flow_id=0), FlowConfig(flow_id=1), FlowConfig(flow_id=2)]
        config = NetworkConfig(flows=flows, flow_weights=[2])
        
        # Should extend flow_weights to match number of flows
        self.assertEqual(config.flow_weights, [2, 1, 1])
    
    def test_post_init_flow_weights_truncation(self):
        """Test that flow_weights are truncated when fewer flows are provided."""
        flows = [FlowConfig(flow_id=0)]
        config = NetworkConfig(flows=flows, flow_weights=[2, 3, 4])
        
        # Should truncate flow_weights to match number of flows
        self.assertEqual(config.flow_weights, [2])
    
    def test_invalid_packet_loss_rate(self):
        """Test validation of packet loss rate."""
        with self.assertRaises(ValueError):
            NetworkConfig(packet_loss_rate=-0.1)
        
        with self.assertRaises(ValueError):
            NetworkConfig(packet_loss_rate=1.1)
    
    def test_invalid_red_thresholds(self):
        """Test validation of RED thresholds."""
        with self.assertRaises(ValueError):
            NetworkConfig(enable_red=True, red_min_threshold=10, red_max_threshold=5)
    
    def test_add_flow(self):
        """Test adding flows."""
        config = NetworkConfig()
        
        # Add a flow
        token_bucket = TokenBucketConfig(rate=1000.0, size=500.0)
        config.add_flow(
            flow_id=2,
            weight=3,
            priority=5,
            congestion_control=CongestionControl.BBR,
            token_bucket=token_bucket
        )
        
        # Check flow was added
        self.assertEqual(len(config.flows), 1)
        flow = config.flows[0]
        self.assertEqual(flow.flow_id, 2)
        self.assertEqual(flow.weight, 3)
        self.assertEqual(flow.priority, 5)
        self.assertEqual(flow.congestion_control, CongestionControl.BBR)
        self.assertEqual(flow.token_bucket, token_bucket)
        
        # Check flow_weights was updated
        self.assertEqual(config.flow_weights, [1, 1, 3])
    
    def test_set_fattree_topology(self):
        """Test setting fat-tree topology."""
        config = NetworkConfig()
        config.set_fattree_topology(k=4)
        
        self.assertEqual(config.topology.topology_type, TopologyType.FATTREE)
        self.assertEqual(config.topology.num_hosts, 16)  # k^3/4
        self.assertEqual(config.topology.num_switches, 20)  # 5*k^2/4
    
    def test_set_internet_topology(self):
        """Test setting Internet Topology Zoo topology."""
        config = NetworkConfig()
        config.set_internet_topology("Abilene")
        
        self.assertEqual(config.topology.topology_type, TopologyType.INTERNET_TOPO_ZOO)
        self.assertEqual(config.topology.topology_file, "ns/topos/internet_topo_zoo/Abilene.graphml")
    
    def test_enable_token_bucket_shaping(self):
        """Test enabling token bucket shaping."""
        config = NetworkConfig()
        config.enable_token_bucket_shaping(rate=2000.0, size=1000.0, peak_rate=4000.0, peak_size=2000.0)
        
        self.assertTrue(config.enable_shaping)
        
        # Should create flows with token bucket
        self.assertEqual(len(config.flows), 2)  # Default flow_weights has 2 elements
        
        for flow in config.flows:
            self.assertIsNotNone(flow.token_bucket)
            self.assertEqual(flow.token_bucket.rate, 2000.0)
            self.assertEqual(flow.token_bucket.size, 1000.0)
            self.assertEqual(flow.token_bucket.peak_rate, 4000.0)
            self.assertEqual(flow.token_bucket.peak_size, 2000.0)
    
    def test_get_ns_config(self):
        """Test conversion to ns.py compatible configuration."""
        config = NetworkConfig(
            source_rate=5000.0,
            flow_weights=[2, 3],
            scheduler=SchedulerType.WFQ,
            packet_loss_rate=0.02,
            debug=True,
            seed=123
        )
        
        # Add a flow
        config.add_flow(flow_id=0, weight=2, congestion_control=CongestionControl.BBR)
        
        ns_config = config.get_ns_config()
        
        # Check basic parameters
        self.assertEqual(ns_config['source_rate'], 5000.0)
        self.assertEqual(ns_config['weights'], [2, 3, 2])
        self.assertEqual(ns_config['scheduler'], 'wfq')
        self.assertEqual(ns_config['packet_loss_rate'], 0.02)
        self.assertTrue(ns_config['debug'])
        self.assertEqual(ns_config['seed'], 123)
        
        # Check topology config
        self.assertIn('topology', ns_config)
        topo = ns_config['topology']
        self.assertEqual(topo['type'], 'simple')
        
        # Check flows config
        self.assertIn('flows', ns_config)
        self.assertEqual(len(ns_config['flows']), 1)
        flow = ns_config['flows'][0]
        self.assertEqual(flow['id'], 0)
        self.assertEqual(flow['weight'], 2)
        self.assertEqual(flow['congestion_control'], 'bbr')


class TestPredefinedConfigs(unittest.TestCase):
    """Test cases for predefined configuration functions."""
    
    def test_get_low_latency_config(self):
        """Test low latency configuration."""
        config = get_low_latency_config()
        
        self.assertEqual(config.source_rate, 100e6)
        self.assertEqual(config.packet_loss_rate, 0.001)
        self.assertEqual(config.congestion_control, CongestionControl.BBR)
        self.assertEqual(config.scheduler, SchedulerType.VIRTUAL_CLOCK)
        self.assertEqual(config.topology.link_delay, 0.001)
    
    def test_get_satellite_config(self):
        """Test satellite configuration."""
        config = get_satellite_config()
        
        self.assertEqual(config.source_rate, 10e6)
        self.assertEqual(config.packet_loss_rate, 0.01)
        self.assertEqual(config.congestion_control, CongestionControl.CUBIC)
        self.assertEqual(config.scheduler, SchedulerType.WFQ)
        self.assertEqual(config.topology.link_delay, 0.3)  # High latency
    
    def test_get_iot_config(self):
        """Test IoT configuration."""
        config = get_iot_config()
        
        self.assertEqual(config.source_rate, 1e6)
        self.assertEqual(config.packet_loss_rate, 0.05)
        self.assertEqual(config.congestion_control, CongestionControl.CUBIC)
        self.assertEqual(config.scheduler, SchedulerType.STATIC_PRIORITY)
        self.assertEqual(config.flow_weights, [2, 1])  # Prioritized flows
        self.assertTrue(config.enable_shaping)


if __name__ == '__main__':
    unittest.main()