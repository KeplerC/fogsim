"""FogSim: A co-simulation framework for robotics and network simulation.

FogSim provides a unified interface for co-simulation between robotics
simulators (Gym, CARLA, Mujoco) and network simulation, allowing researchers
to study the effects of network latency and bandwidth constraints on robotic systems.
"""

# Main environment interface
from .env import Env

# Handler-based architecture
from .handlers import BaseHandler, GymHandler, CarlaHandler, MujocoHandler

# Network simulation
from .network import NetworkConfig, NSPyNetworkSimulator
from .network import TopologyType, CongestionControl, SchedulerType
from .network import get_low_latency_config, get_satellite_config, get_iot_config

# Time backend system
from .time_backend import (
    SimulationMode,
    UnifiedTimeManager,
    TimeBackend,
    VirtualTimeBackend,
    SimulatedNetworkTimeBackend,
    RealNetworkTimeBackend
)

# Network control
from .network_control import (
    NetworkControlManager,
    NetworkConfig as NetworkControlConfig,
    TCController
)

# Message passing
from .message_passing import (
    MessageBus,
    TimedMessage,
    SimpleMessageHandler
)

# Real network support
from .real_network import (
    RealNetworkManager,
    LatencyMeasurer,
    NetworkMeasurement,
    create_real_network_config
)
from .real_network_client import (
    RealNetworkClient,
    RealNetworkMessageHandler,
    test_real_network_connection
)

# Legacy co-simulators (for backward compatibility)
from .base import BaseCoSimulator
from .environment.gym_co_simulator import GymCoSimulator
from .environment.carla_co_simulator import CarlaCoSimulator

__version__ = "0.1.0"

__all__ = [
    # Main interface
    'Env',
    
    # Handlers
    'BaseHandler',
    'GymHandler', 
    'CarlaHandler',
    'MujocoHandler',
    
    # Network simulation
    'NetworkConfig',
    'NSPyNetworkSimulator',
    'TopologyType',
    'CongestionControl',
    'SchedulerType',
    'get_low_latency_config',
    'get_satellite_config',
    'get_iot_config',
    
    # Time backend
    'SimulationMode',
    'UnifiedTimeManager',
    'TimeBackend',
    'VirtualTimeBackend',
    'SimulatedNetworkTimeBackend',
    'RealNetworkTimeBackend',
    
    # Network control
    'NetworkControlManager',
    'NetworkControlConfig',
    'TCController',
    
    # Message passing
    'MessageBus',
    'TimedMessage',
    'SimpleMessageHandler',
    
    # Real network
    'RealNetworkManager',
    'LatencyMeasurer',
    'NetworkMeasurement',
    'create_real_network_config',
    'RealNetworkClient',
    'RealNetworkMessageHandler',
    'test_real_network_connection',
    
    # Legacy (backward compatibility)
    'BaseCoSimulator',
    'GymCoSimulator',
    'CarlaCoSimulator',
] 