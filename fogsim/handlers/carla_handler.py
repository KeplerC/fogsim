"""CARLA simulator handler for FogSim.

This module provides a handler implementation for CARLA autonomous driving
simulator, wrapping it to work with the FogSim framework.
"""

from typing import Any, Dict, Optional, List, Tuple
import numpy as np
import logging
import time

from .base_handler import BaseHandler

carla = None
try:
    import carla
except ImportError:
    pass

logger = logging.getLogger(__name__)


class CarlaHandler(BaseHandler):
    """Handler for CARLA autonomous driving simulator.
    
    This handler wraps CARLA simulator to work with the FogSim framework,
    providing a consistent interface for simulation control.
    
    Args:
        host: CARLA server host (default: 'localhost')
        port: CARLA server port (default: 2000)
        timeout: Connection timeout in seconds (default: 10.0)
        synchronous: Whether to run in synchronous mode (default: True)
        fixed_delta_seconds: Fixed time step for synchronous mode (default: 0.05)
        render_mode: Rendering mode ('camera', 'spectator', None)
        vehicle_filter: Filter for spawning vehicles (default: 'vehicle.*')
        map_name: Name of the map to load (optional)
    """
    
    def __init__(self,
                 host: str = 'localhost',
                 port: int = 2000,
                 timeout: float = 10.0,
                 synchronous: bool = True,
                 fixed_delta_seconds: float = 0.05,
                 render_mode: Optional[str] = None,
                 vehicle_filter: str = 'vehicle.*',
                 map_name: Optional[str] = None):
        """Initialize the CARLA handler."""
        if carla is None:
            raise ImportError(
                "CARLA Python API is not installed. Please install it with: "
                "pip install 'fogsim[carla]' or follow CARLA installation guide"
            )
        
        self.host = host
        self.port = port
        self.timeout = timeout
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds
        self.render_mode = render_mode
        self.vehicle_filter = vehicle_filter
        self.map_name = map_name
        
        self._client = None
        self._world = None
        self._vehicle = None
        self._sensors = {}
        self._sensor_data = {}
        self._launched = False
        self._original_settings = None
        self._spawn_points = []
        self._last_observation = None
        self._last_control = None
        self._step_count = 0
        self._episode_count = 0
    
    def launch(self) -> None:
        """Launch CARLA client and setup the simulation."""
        if self._launched:
            logger.warning("CARLA handler already launched")
            return
        
        try:
            # Connect to CARLA server
            logger.info(f"Connecting to CARLA server at {self.host}:{self.port}")
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout)
            
            # Load map if specified
            if self.map_name:
                logger.info(f"Loading map: {self.map_name}")
                self._world = self._client.load_world(self.map_name)
            else:
                self._world = self._client.get_world()
            
            # Store original settings
            self._original_settings = self._world.get_settings()
            
            # Setup synchronous mode
            if self.synchronous:
                settings = self._world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
                self._world.apply_settings(settings)
                logger.info(f"Synchronous mode enabled with delta={self.fixed_delta_seconds}")
            
            # Get spawn points
            self._spawn_points = self._world.get_map().get_spawn_points()
            
            # Spawn vehicle
            self._spawn_vehicle()
            
            # Setup sensors if rendering
            if self.render_mode == 'camera':
                self._setup_camera()
            
            self._launched = True
            logger.info("CARLA handler launched successfully")
            
        except Exception as e:
            logger.error(f"Failed to launch CARLA: {e}")
            raise
    
    def _spawn_vehicle(self) -> None:
        """Spawn a vehicle in the world."""
        blueprint_library = self._world.get_blueprint_library()
        vehicle_blueprints = blueprint_library.filter(self.vehicle_filter)
        
        if not vehicle_blueprints:
            raise RuntimeError(f"No vehicles found with filter: {self.vehicle_filter}")
        
        # Choose a random vehicle blueprint
        vehicle_bp = np.random.choice(vehicle_blueprints)
        
        # Choose a random spawn point
        spawn_point = np.random.choice(self._spawn_points)
        
        # Spawn the vehicle
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)
        logger.info(f"Spawned vehicle: {vehicle_bp.id} at {spawn_point.location}")
        
        # Enable autopilot initially
        self._vehicle.set_autopilot(False)
    
    def _setup_camera(self) -> None:
        """Setup camera sensor for rendering."""
        blueprint_library = self._world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Attach camera to vehicle
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = self._world.spawn_actor(camera_bp, camera_transform, attach_to=self._vehicle)
        
        # Setup camera callback
        camera.listen(lambda image: self._on_camera_data(image))
        self._sensors['camera'] = camera
        logger.info("Camera sensor setup complete")
    
    def _on_camera_data(self, image: Any) -> None:
        """Callback for camera data."""
        # Convert CARLA image to numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3]  # Remove alpha channel
        self._sensor_data['camera'] = array
    
    def set_states(self, states: Optional[Dict[str, Any]] = None,
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states.
        
        Args:
            states: Optional state dictionary containing:
                - 'transform': Vehicle transform
                - 'velocity': Vehicle velocity
            action: Control action [throttle, steer, brake]
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if states is None:
            # Reset vehicle to random spawn point
            self._reset_vehicle()
        else:
            # Set vehicle transform if provided
            if 'transform' in states:
                self._vehicle.set_transform(states['transform'])
            
            # Set vehicle velocity if provided
            if 'velocity' in states:
                self._vehicle.set_target_velocity(states['velocity'])
        
        if action is not None:
            # Apply control action
            control = carla.VehicleControl()
            control.throttle = float(np.clip(action[0], 0, 1))
            control.steer = float(np.clip(action[1], -1, 1))
            if len(action) > 2:
                control.brake = float(np.clip(action[2], 0, 1))
            
            self._vehicle.apply_control(control)
            self._last_control = control
    
    def _reset_vehicle(self) -> None:
        """Reset vehicle to a random spawn point."""
        spawn_point = np.random.choice(self._spawn_points)
        self._vehicle.set_transform(spawn_point)
        self._vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self._step_count = 0
        self._episode_count += 1
        logger.info(f"Vehicle reset to spawn point: {spawn_point.location}")
    
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states.
        
        Returns:
            Dictionary containing vehicle state information
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        transform = self._vehicle.get_transform()
        velocity = self._vehicle.get_velocity()
        acceleration = self._vehicle.get_acceleration()
        angular_velocity = self._vehicle.get_angular_velocity()
        
        # Create observation vector
        observation = np.array([
            transform.location.x,
            transform.location.y,
            transform.location.z,
            transform.rotation.pitch,
            transform.rotation.yaw,
            transform.rotation.roll,
            velocity.x,
            velocity.y,
            velocity.z,
            angular_velocity.x,
            angular_velocity.y,
            angular_velocity.z
        ])
        
        self._last_observation = observation
        
        return {
            'observation': observation,
            'transform': transform,
            'velocity': velocity,
            'acceleration': acceleration,
            'angular_velocity': angular_velocity,
            'location': [transform.location.x, transform.location.y, transform.location.z],
            'rotation': [transform.rotation.pitch, transform.rotation.yaw, transform.rotation.roll],
            'step_count': self._step_count,
            'episode_count': self._episode_count
        }
    
    def step(self) -> None:
        """Step the simulation forward."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.synchronous:
            self._world.tick()
        else:
            # In asynchronous mode, just wait for a small time
            time.sleep(self.fixed_delta_seconds)
        
        self._step_count += 1
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state.
        
        Returns:
            Camera image if render_mode is 'camera', None otherwise
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.render_mode == 'camera' and 'camera' in self._sensor_data:
            return self._sensor_data['camera']
        
        return None
    
    def close(self) -> None:
        """Clean up CARLA resources."""
        if self._launched:
            # Destroy sensors
            for sensor in self._sensors.values():
                sensor.destroy()
            self._sensors.clear()
            
            # Destroy vehicle
            if self._vehicle is not None:
                self._vehicle.destroy()
                self._vehicle = None
            
            # Restore original settings
            if self._original_settings is not None:
                self._world.apply_settings(self._original_settings)
            
            self._launched = False
            logger.info("CARLA handler closed")
    
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata.
        
        Returns:
            Dictionary containing CARLA-specific information
        """
        if not self._launched:
            return {
                'host': self.host,
                'port': self.port,
                'launched': False
            }
        
        return {
            'host': self.host,
            'port': self.port,
            'launched': True,
            'map_name': self._world.get_map().name,
            'vehicle_type': self._vehicle.type_id if self._vehicle else None,
            'synchronous': self.synchronous,
            'fixed_delta_seconds': self.fixed_delta_seconds,
            'render_mode': self.render_mode,
            'num_spawn_points': len(self._spawn_points),
            'weather': str(self._world.get_weather())
        }
    
    @property
    def is_launched(self) -> bool:
        """Check if the handler has been launched."""
        return self._launched