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
    
    # Legacy (backward compatibility)
    'BaseCoSimulator',
    'GymCoSimulator',
    'CarlaCoSimulator',
] 