"""
FogSim: Simplified Co-simulation Framework for Robotics and Network Simulation

FogSim provides three simulation modes as described in CLAUDE.md:
1. Virtual timeline (VIRTUAL) - Decoupled from wallclock for scalability
2. Real clock + simulated network (SIMULATED_NET) - High frame rate with network simulation  
3. Real clock + real network (REAL_NET) - Shows sim-to-real gap

Refactored for simplicity and breaking changes from legacy API.
"""

# Core interface
from .core import FogSim, SimulationMode

# Handlers (unchanged)
from .handlers import BaseHandler, GymHandler, CarlaHandler, MujocoHandler

# Network components (simplified)
from .network import NetworkConfig, NSPyNetworkSimulator
from .network.real_network import RealNetworkTransport

# Clock components (new)
from .clock import VirtualClock, RealClock

# Simple messages
from .messages import Message, SimulationMessage, create_step_message

# Legacy environment support (minimal)
from .env import Env

__version__ = "0.2.0"  # Breaking changes version bump

__all__ = [
    # Core interface
    'FogSim',
    'SimulationMode',
    
    # Handlers  
    'BaseHandler',
    'GymHandler', 
    'CarlaHandler',
    'MujocoHandler',
    
    # Network
    'NetworkConfig',
    'NSPyNetworkSimulator', 
    'RealNetworkTransport',
    
    # Clock
    'VirtualClock',
    'RealClock',
    
    # Messages
    'Message',
    'SimulationMessage',
    'create_step_message',
    
    # Legacy support
    'Env',
]