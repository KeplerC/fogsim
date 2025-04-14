from abc import ABC
from abc import abstractmethod
import argparse
import math
import os
import time

import carla
import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.stats import norm

from configs import *
from utils import *
from cloudsim import GymCoSimulator, CarlaCoSimulator
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("carla_cosimulator.log")  # Save logs to file
    ]
)

logger = logging.getLogger(__name__)

class CarlaLatencySimulator:
    """
    A gym-like interface for CARLA simulator with latency simulation.
    Implements the standard gym methods: step(), reset(), render().
    """
    
    def __init__(self, config, output_dir=None):
        """
        Initialize the CARLA simulator with latency.
        
        Args:
            config (dict): Configuration dictionary
            output_dir (str): Directory to store results
        """
        self.config = config
        self.output_dir = output_dir
        
        # Connection to CARLA
        self.client = None
        self.world = None
        self.original_settings = None
        
        # Actors
        self.ego_vehicle = None
        self.obstacle_vehicle = None
        self.camera = None
        self.collision_sensor = None
        
        # Simulation state
        self.has_collided = False
        self.frame_queue = []
        self.tick = 0
        
        # Tracking
        self.obstacle_buffer = []
        
        # Connect to CARLA and initialize
        self._connect_to_carla()
        
    def _connect_to_carla(self):
        """Connect to CARLA and set up the simulation environment."""
        self.client = carla.Client(self.config['simulation']['host'],
                              self.config['simulation']['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town03")

        # Set synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.config['simulation']['delta_seconds']
        settings.synchronous_mode = True
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        
    def reset(self):
        """
        Reset the simulation to initial state.
        
        Returns:
            observation: Initial observation (usually a camera frame)
        """
        # Clean up any existing actors
        self._clean_up_actors()
        
        # Reset simulation state
        self.has_collided = False
        self.frame_queue = []
        self.obstacle_buffer = []
        
        # Spawn the vehicles and sensors
        self._spawn_actors()
        
        # Tick the world to get initial observation
        self.world.tick()
        
        # Return the initial observation
        return self._get_observation()
    
    def step(self, action=None):
        """
        Take a step in the simulation.
        
        Args:
            action: Boolean indicating whether to apply emergency brake
        
        Returns:
            observation: Current observation
        """
        if action is None:
            brake = False
        else:
            brake = action
            
        # Generate and apply ego vehicle control based on current tick and brake flag
        if brake:
            # Emergency brake
            ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        else:
            # Normal driving based on config
            ego_control = carla.VehicleControl()
            
            if self.tick < self.config['ego_vehicle']['go_straight_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['straight']
                ego_control.steer = self.config['ego_vehicle']['steer']['straight'] if 'straight' in self.config['ego_vehicle']['steer'] else 0.0
            elif self.tick < self.config['ego_vehicle']['go_straight_ticks'] + self.config['ego_vehicle']['turn_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['turn']
                ego_control.steer = self.config['ego_vehicle']['steer']['turn']
            else:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = self.config['ego_vehicle']['steer']['after_turn'] if 'after_turn' in self.config['ego_vehicle']['steer'] else 0.0
        
        # Apply the calculated control
        self.ego_vehicle.apply_control(ego_control)
        
        # Tick the world
        self.world.tick()
        self.tick += 1
        
        # Apply obstacle controls
        self._apply_obstacle_controls()
        
        # Get observation
        observation = self._get_observation()

        return observation
    
    def _apply_obstacle_controls(self):
        obstacle_control = carla.VehicleControl()

        if self.tick < self.config['obstacle_vehicle']['go_straight_ticks']:
            # Initial straight phase
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['straight']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['straight']
        elif self.tick < self.config['obstacle_vehicle']['go_straight_ticks'] + self.config[
                'obstacle_vehicle']['turn_ticks']:
            # Turning phase
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['turn']
        else:
            # After turn straight phase
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['after_turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['after_turn']

        self.obstacle_vehicle.apply_control(obstacle_control)
            
    def render(self, mode='rgb_array'):
        """
        Render the current simulation state.
        
        Args:
            mode (str): Rendering mode ('rgb_array')
            
        Returns:
            numpy.ndarray: Camera frame if available, otherwise None
        """
        if self.frame_queue:
            return self.frame_queue[-1]
        return None
    
    def close(self):
        """Clean up resources and reset CARLA settings."""
        self._clean_up_actors()
        
        if self.world is not None:
            self.world.apply_settings(self.original_settings)
            
    def _spawn_actors(self):
        """Spawn vehicles and sensors in the simulation."""
        blueprint_library = self.world.get_blueprint_library()

        # Get spawn points
        spawn_points = self.world.get_map().get_spawn_points()

        # Setup ego vehicle spawn point
        ego_spawn_point = spawn_points[0]
        ego_spawn_point.location.x += self.config['ego_vehicle']['spawn_offset']['x']
        ego_spawn_point.location.y += self.config['ego_vehicle']['spawn_offset']['y']
        ego_spawn_point.rotation.yaw += self.config['ego_vehicle']['spawn_offset']['yaw']

        obstacle_spawn_point = spawn_points[1]
        obstacle_spawn_point.location.x = ego_spawn_point.location.x + self.config[
            'obstacle_vehicle']['spawn_offset']['x']
        obstacle_spawn_point.location.y = ego_spawn_point.location.y + self.config[
            'obstacle_vehicle']['spawn_offset']['y']
        obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + self.config[
            'obstacle_vehicle']['spawn_offset']['yaw']

        # Spawn both vehicles
        ego_bp = blueprint_library.find(self.config['ego_vehicle']['model'])
        ego_bp.set_attribute('role_name', 'ego')
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn_point)

        obstacle_bp = blueprint_library.find(self.config['obstacle_vehicle']['model'])
        obstacle_bp.set_attribute('role_name', 'obstacle')
        self.obstacle_vehicle = self.world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

        # Attach camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.config['video']['width']))
        camera_bp.set_attribute('image_size_y', str(self.config['video']['height']))
        camera_bp.set_attribute('fov', self.config['camera']['fov'])

        camera_transform = carla.Transform(
            carla.Location(
                x=ego_spawn_point.location.x + self.config['camera']['offset']['x'],
                y=ego_spawn_point.location.y + self.config['camera']['offset']['y'],
                z=self.config['camera']['height']), carla.Rotation(pitch=-90))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=None)

        # Setup camera callback
        def camera_callback(image):
            image.convert(carla.ColorConverter.Raw)
            img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
            img_array = img_array.reshape((image.height, image.width, 4))
            frame_bgr = img_array[:, :, :3].copy()
            self.frame_queue.append(frame_bgr)

        self.camera.listen(camera_callback)

        # Add collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp,
                                             carla.Transform(),
                                             attach_to=self.ego_vehicle)

        def collision_callback(event):
            self.has_collided = True

        self.collision_sensor.listen(collision_callback)
        
    def _clean_up_actors(self):
        """Clean up all actors in the simulation."""
        if self.collision_sensor is not None:
            self.collision_sensor.stop()
            self.collision_sensor.destroy()
            self.collision_sensor = None

        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            self.camera = None

        if self.ego_vehicle is not None:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
            
        if self.obstacle_vehicle is not None:
            self.obstacle_vehicle.destroy()
            self.obstacle_vehicle = None
    
    def _get_observation(self):
        """Get the current observation from the simulation."""
        # Make sure vehicles are initialized
        if self.ego_vehicle is None or self.obstacle_vehicle is None:
            return None
        
        # Convert transform to numpy array with position and rotation
        ego_transform = self.ego_vehicle.get_transform()
        obstacle_transform = self.obstacle_vehicle.get_transform()
        
        # Get position data
        ego_pos = np.array([
            ego_transform.location.x,
            ego_transform.location.y,
            ego_transform.location.z
        ])
        
        obstacle_pos = np.array([
            obstacle_transform.location.x,
            obstacle_transform.location.y,
            obstacle_transform.location.z
        ])
        
        # Get rotation data (yaw, pitch, roll in degrees)
        ego_rot = np.array([
            ego_transform.rotation.yaw,
            ego_transform.rotation.pitch,
            ego_transform.rotation.roll
        ])
        
        obstacle_rot = np.array([
            obstacle_transform.rotation.yaw,
            obstacle_transform.rotation.pitch,
            obstacle_transform.rotation.roll
        ])
        
        # Construct observation with position, rotation and tick
        current_observation = (
            ego_pos,
            obstacle_pos,
            ego_rot,
            obstacle_rot, 
            self.tick
        )
        
        return current_observation
    

def main():
    parser = argparse.ArgumentParser(
        description='Run CARLA simulation with configurable parameters')
    parser.add_argument('--config_type',
                        type=str,
                        choices=['right_turn', 'left_turn', 'merge'],
                        default='right_turn',
                        help='Type of configuration to use')
    parser.add_argument('--emergency_brake_threshold',
                        type=float,
                        default=1.1,
                        help='Threshold for emergency braking')
    parser.add_argument('--output_dir',
                        type=str,
                        default='./results',
                        help='Directory to store results')

    args = parser.parse_args()

    network_sim = NSPyNetworkSimulator(
        source_rate=100000000.0,  # 10 Mbps
        weights=[1, 2],       # Weight client->server flows lower than server->client
    )
    # Select configuration based on argument
    config_map = {
        'right_turn': unprotected_right_turn_config,
        'left_turn': unprotected_left_turn_config,
        'merge': opposite_direction_merge_config
    }

    base_config = config_map[args.config_type]

    # Update emergency brake threshold
    base_config['simulation'][
        'emergency_brake_threshold'] = args.emergency_brake_threshold

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run in gym-like mode for interactive use or testing
    simulator = CarlaLatencySimulator(base_config, args.output_dir)
    
    # Get initial observation
    observation = simulator.reset()
    
    # Create the relative position tracker with time step and debug setting
    rel_tracker = EKFObstacleTracker(
        simulator.ego_vehicle,
        simulator.obstacle_vehicle,
        dt=base_config['simulation']['delta_seconds']
    )
    
    # Initialize co-simulator
    co_sim = CarlaCoSimulator(network_sim, simulator, timestep=0.01)

    brake = False
    
    logger.info(f"Starting simulation with {base_config['simulation']['prediction_steps']} prediction steps")
    logger.info(f"Emergency brake threshold: {base_config['simulation']['emergency_brake_threshold']}")

    total_steps = base_config['ego_vehicle']['go_straight_ticks'] + base_config['ego_vehicle']['turn_ticks'] + base_config['ego_vehicle']['after_turn_ticks']
    # Run for specified number of steps
    for cur_step in range(total_steps):
        
        # Step the simulation with brake flag
        observation = co_sim.step(brake)
        
        # Reset brake flag
        brake = False
        
        # Predict future positions and calculate collision probability
        if observation is not None:
            ego_pos = observation[0]
            obstacle_pos = observation[1]
            ego_rot = observation[2]
            obstacle_rot = observation[3]
            logger.warning(f"Current latency is {cur_step - observation[4]}")
            
            tick = cur_step
            
            # relative x, y, yaw
            rel_x = obstacle_pos[0] - ego_pos[0]
            rel_y = obstacle_pos[1] - ego_pos[1]
            rel_yaw_deg = obstacle_rot[0] - ego_rot[0] 
            rel_yaw_rad = rel_yaw_deg * np.pi / 180.0
                        
            # Update the tracker with the observation data
            rel_tracker.update((rel_x, rel_y, rel_yaw_rad), tick)
            
            # Predict future positions
            predicted_positions = rel_tracker.predict_future_position(1000)
            
            
            # Calculate collision probabilities
            ego_trajectory = np.zeros((total_steps, 3))

            max_collision_prob, collision_time, collision_probabilities = calculate_collision_probabilities(
                rel_tracker, predicted_positions, ego_trajectory, tick)
            logger.info(f"Step {cur_step}: Collision probability: {max_collision_prob:.4f}")
            
            # Apply emergency brake if collision probability exceeds threshold
            if max_collision_prob > base_config['simulation']['emergency_brake_threshold']:
                logger.info(f"Step {cur_step}: EMERGENCY BRAKE ACTIVATED")
                brake = True
                
            if simulator.has_collided:
                logger.info(f"Step {cur_step}: Collision detected")
                break
        else:
            logger.info(f"Step {cur_step}: No observation received from simulator")

    logger.info("Simulation completed")
    


if __name__ == '__main__':
    main()
