"""Network configuration classes for FogSim.

This module provides configuration classes for setting up network simulation
parameters, exposing the rich capabilities of the ns.py network simulator.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class TopologyType(Enum):
    """Supported network topology types."""
    SIMPLE = "simple"
    FATTREE = "fattree"
    CUSTOM = "custom"
    INTERNET_TOPO_ZOO = "internet_topo_zoo"


class CongestionControl(Enum):
    """Supported congestion control algorithms."""
    CUBIC = "cubic"
    BBR = "bbr"
    RENO = "reno"
    VEGAS = "vegas"


class SchedulerType(Enum):
    """Supported packet scheduling algorithms."""
    VIRTUAL_CLOCK = "virtual_clock"
    WFQ = "wfq"  # Weighted Fair Queuing
    DRR = "drr"  # Deficit Round Robin
    FIFO = "fifo"
    STATIC_PRIORITY = "sp"


@dataclass
class TokenBucketConfig:
    """Configuration for token bucket traffic shaping.
    
    Args:
        rate: Token generation rate (bytes/second)
        size: Bucket size (bytes)
        peak_rate: Peak rate for two-rate token bucket (optional)
        peak_size: Peak bucket size for two-rate token bucket (optional)
    """
    rate: float
    size: float
    peak_rate: Optional[float] = None
    peak_size: Optional[float] = None


@dataclass
class FlowConfig:
    """Configuration for individual network flows.
    
    Args:
        flow_id: Unique identifier for the flow
        weight: Weight for scheduling (used by WFQ, DRR)
        priority: Priority level (used by static priority scheduler)
        congestion_control: Congestion control algorithm for this flow
        token_bucket: Token bucket configuration for traffic shaping
    """
    flow_id: int
    weight: int = 1
    priority: int = 0
    congestion_control: Optional[CongestionControl] = None
    token_bucket: Optional[TokenBucketConfig] = None


@dataclass
class TopologyConfig:
    """Configuration for network topology.
    
    Args:
        topology_type: Type of topology to use
        num_hosts: Number of hosts (for fattree and custom topologies)
        num_switches: Number of switches (for custom topologies)
        link_bandwidth: Default link bandwidth (bytes/second)
        link_delay: Default link delay (seconds)
        topology_file: Path to topology file (for internet_topo_zoo)
        custom_topology: Custom topology definition
    """
    topology_type: TopologyType = TopologyType.SIMPLE
    num_hosts: int = 4
    num_switches: int = 1
    link_bandwidth: float = 1e6  # 1 Mbps
    link_delay: float = 0.001  # 1ms
    topology_file: Optional[str] = None
    custom_topology: Optional[Dict[str, Any]] = None


@dataclass
class NetworkConfig:
    """Main network simulation configuration.
    
    This class exposes the rich configuration options available in ns.py,
    allowing users to configure various aspects of network simulation.
    
    Args:
        topology: Network topology configuration
        scheduler: Packet scheduling algorithm
        source_rate: Default source rate for network simulator (bytes/second)
        flow_weights: Default weights for flows
        flows: Per-flow configurations
        packet_loss_rate: Packet loss rate (0.0 to 1.0)
        enable_shaping: Whether to enable traffic shaping
        congestion_control: Default congestion control algorithm
        buffer_size: Buffer size for switches/ports (bytes)
        enable_red: Whether to enable RED (Random Early Detection)
        red_min_threshold: RED minimum threshold
        red_max_threshold: RED maximum threshold
        red_drop_probability: RED maximum drop probability
        simulation_time: Maximum simulation time (seconds)
        debug: Enable debug logging
        seed: Random seed for reproducibility
    """
    # Topology configuration
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    
    # Scheduling configuration
    scheduler: SchedulerType = SchedulerType.VIRTUAL_CLOCK
    source_rate: float = 4600.0  # bytes/second
    flow_weights: List[int] = field(default_factory=lambda: [1, 1])
    
    # Per-flow configurations
    flows: List[FlowConfig] = field(default_factory=list)
    
    # Network characteristics
    packet_loss_rate: float = 0.0
    buffer_size: int = 100000  # bytes
    
    # Traffic shaping
    enable_shaping: bool = False
    
    # Congestion control
    congestion_control: CongestionControl = CongestionControl.CUBIC
    
    # RED configuration
    enable_red: bool = False
    red_min_threshold: int = 5
    red_max_threshold: int = 15
    red_drop_probability: float = 0.02
    
    # Simulation parameters
    simulation_time: float = 100.0
    debug: bool = False
    seed: Optional[int] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate packet loss rate
        if not 0.0 <= self.packet_loss_rate <= 1.0:
            raise ValueError("packet_loss_rate must be between 0.0 and 1.0")
        
        # Validate RED thresholds
        if self.enable_red and self.red_min_threshold >= self.red_max_threshold:
            raise ValueError("red_min_threshold must be less than red_max_threshold")
    
    def add_flow(self, flow_id: int, weight: int = 1, priority: int = 0,
                 congestion_control: Optional[CongestionControl] = None,
                 token_bucket: Optional[TokenBucketConfig] = None) -> None:
        """Add a flow configuration.
        
        Args:
            flow_id: Unique identifier for the flow
            weight: Weight for scheduling
            priority: Priority level
            congestion_control: Congestion control algorithm
            token_bucket: Token bucket configuration
        """
        flow_config = FlowConfig(
            flow_id=flow_id,
            weight=weight,
            priority=priority,
            congestion_control=congestion_control,
            token_bucket=token_bucket
        )
        self.flows.append(flow_config)
        
        # Always append the new flow's weight to flow_weights
        self.flow_weights.append(weight)
    
    def set_fattree_topology(self, k: int = 4) -> None:
        """Configure a fat-tree topology.
        
        Args:
            k: Fat-tree parameter (number of ports per switch)
        """
        self.topology = TopologyConfig(
            topology_type=TopologyType.FATTREE,
            num_hosts=k**3 // 4,
            num_switches=5 * k**2 // 4
        )
    
    def set_internet_topology(self, topology_name: str) -> None:
        """Configure an Internet Topology Zoo topology.
        
        Args:
            topology_name: Name of the topology (e.g., 'Abilene', 'Geant2012')
        """
        self.topology = TopologyConfig(
            topology_type=TopologyType.INTERNET_TOPO_ZOO,
            topology_file=f"ns/topos/internet_topo_zoo/{topology_name}.graphml"
        )
    
    def enable_token_bucket_shaping(self, rate: float, size: float,
                                   peak_rate: Optional[float] = None,
                                   peak_size: Optional[float] = None) -> None:
        """Enable token bucket traffic shaping.
        
        Args:
            rate: Token generation rate (bytes/second)
            size: Bucket size (bytes)
            peak_rate: Peak rate for two-rate token bucket
            peak_size: Peak bucket size for two-rate token bucket
        """
        self.enable_shaping = True
        
        # Add token bucket to all flows or create default flow
        token_bucket = TokenBucketConfig(
            rate=rate, size=size, peak_rate=peak_rate, peak_size=peak_size
        )
        
        if not self.flows:
            # Create default flows
            for i, weight in enumerate(self.flow_weights):
                self.add_flow(i, weight=weight, token_bucket=token_bucket)
        else:
            # Add to existing flows
            for flow in self.flows:
                flow.token_bucket = token_bucket
    
    def get_ns_config(self) -> Dict[str, Any]:
        """Convert to ns.py compatible configuration.
        
        Returns:
            Dictionary with ns.py configuration parameters
        """
        config = {
            'source_rate': self.source_rate,
            'weights': self.flow_weights,
            'debug': self.debug
        }
        
        # Add scheduler-specific config
        if self.scheduler == SchedulerType.VIRTUAL_CLOCK:
            config['scheduler'] = 'virtual_clock'
        elif self.scheduler == SchedulerType.WFQ:
            config['scheduler'] = 'wfq'
        elif self.scheduler == SchedulerType.DRR:
            config['scheduler'] = 'drr'
        
        # Add topology config
        config['topology'] = {
            'type': self.topology.topology_type.value,
            'num_hosts': self.topology.num_hosts,
            'num_switches': self.topology.num_switches,
            'link_bandwidth': self.topology.link_bandwidth,
            'link_delay': self.topology.link_delay
        }
        
        if self.topology.topology_file:
            config['topology']['file'] = self.topology.topology_file
        
        # Add flow configs
        if self.flows:
            config['flows'] = []
            for flow in self.flows:
                flow_config = {
                    'id': flow.flow_id,
                    'weight': flow.weight,
                    'priority': flow.priority
                }
                
                if flow.congestion_control:
                    flow_config['congestion_control'] = flow.congestion_control.value
                
                if flow.token_bucket:
                    flow_config['token_bucket'] = {
                        'rate': flow.token_bucket.rate,
                        'size': flow.token_bucket.size
                    }
                    if flow.token_bucket.peak_rate:
                        flow_config['token_bucket']['peak_rate'] = flow.token_bucket.peak_rate
                        flow_config['token_bucket']['peak_size'] = flow.token_bucket.peak_size
                
                config['flows'].append(flow_config)
        
        # Add other parameters
        config.update({
            'packet_loss_rate': self.packet_loss_rate,
            'buffer_size': self.buffer_size,
            'enable_red': self.enable_red,
            'simulation_time': self.simulation_time
        })
        
        if self.enable_red:
            config.update({
                'red_min_threshold': self.red_min_threshold,
                'red_max_threshold': self.red_max_threshold,
                'red_drop_probability': self.red_drop_probability
            })
        
        if self.seed is not None:
            config['seed'] = self.seed
        
        return config


# Predefined configurations for common scenarios
def get_low_latency_config() -> NetworkConfig:
    """Get configuration for low-latency networks (e.g., 5G, edge computing)."""
    config = NetworkConfig(
        source_rate=100e6,  # 100 Mbps
        flow_weights=[1, 1],
        packet_loss_rate=0.001,  # 0.1% loss
        congestion_control=CongestionControl.BBR,
        scheduler=SchedulerType.VIRTUAL_CLOCK
    )
    
    config.topology = TopologyConfig(
        topology_type=TopologyType.SIMPLE,
        link_bandwidth=100e6,  # 100 Mbps
        link_delay=0.001  # 1ms
    )
    
    return config


def get_satellite_config() -> NetworkConfig:
    """Get configuration for satellite networks (high latency, variable bandwidth)."""
    config = NetworkConfig(
        source_rate=10e6,  # 10 Mbps
        flow_weights=[1, 1],
        packet_loss_rate=0.01,  # 1% loss
        congestion_control=CongestionControl.CUBIC,
        scheduler=SchedulerType.WFQ
    )
    
    config.topology = TopologyConfig(
        topology_type=TopologyType.SIMPLE,
        link_bandwidth=10e6,  # 10 Mbps
        link_delay=0.3  # 300ms (geostationary satellite)
    )
    
    return config


def get_iot_config() -> NetworkConfig:
    """Get configuration for IoT networks (low bandwidth, variable latency)."""
    config = NetworkConfig(
        source_rate=1e6,  # 1 Mbps
        flow_weights=[2, 1],  # Prioritize sensor data
        packet_loss_rate=0.05,  # 5% loss
        congestion_control=CongestionControl.CUBIC,
        scheduler=SchedulerType.STATIC_PRIORITY
    )
    
    config.topology = TopologyConfig(
        topology_type=TopologyType.SIMPLE,
        link_bandwidth=1e6,  # 1 Mbps
        link_delay=0.1  # 100ms
    )
    
    # Enable token bucket for rate limiting
    config.enable_token_bucket_shaping(rate=100e3, size=10e3)  # 100 kbps, 10KB bucket
    
    return config