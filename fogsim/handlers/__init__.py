"""Handler-based architecture for FogSim.

This module provides a handler-based interface that aligns with standard
Mujoco/Roboverse interfaces, allowing for seamless integration with various
robotics simulators.
"""

from .base_handler import BaseHandler

# Import handlers conditionally to avoid import errors
# when optional dependencies are not installed
_available_handlers = ['BaseHandler']

try:
    from .gym_handler import GymHandler
    _available_handlers.append('GymHandler')
except ImportError:
    GymHandler = None

try:
    from .carla_handler import CarlaHandler
    _available_handlers.append('CarlaHandler')
except ImportError:
    CarlaHandler = None

try:
    from .mujoco_handler import MujocoHandler
    _available_handlers.append('MujocoHandler')
except ImportError:
    MujocoHandler = None

__all__ = _available_handlers