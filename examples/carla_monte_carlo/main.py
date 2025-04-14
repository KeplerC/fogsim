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
class RelativePositionTracker:
    """Tracker for predicting relative positions between ego and obstacle vehicles using EKF"""
    def __init__(self, dt=0.1, debug=False):
        """
        Initialize a relative position tracker with Extended Kalman Filter
        
        Args:
            dt (float): Time step interval in seconds
            debug (bool): Enable debug messages
        """
        self.dt = dt
        self.debug = debug
        
        # Initialize EKF
        self.ekf = KalmanFilter(dim_x=6, dim_z=3)
        
        # State vector [x, y, yaw, vx, vy, yaw_rate]
        self.ekf.x = np.zeros(6)
        
        # State transition matrix
        self.ekf.F = np.array([
            [1, 0, 0, self.dt, 0, 0],
            [0, 1, 0, 0, self.dt, 0],
            [0, 0, 1, 0, 0, self.dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.ekf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        self.ekf.R = np.eye(3) * 0.1
        
        # Process noise
        self.ekf.Q = np.eye(6) * 0.01
        self.ekf.Q[3:, 3:] *= 10  # Higher process noise for velocity components
        
        # Initial covariance
        self.ekf.P = np.eye(6) * 1.0
        
        # Last update time
        self.last_update_time = 0
        
        # Previous position data for calculating velocities
        self.prev_ego_pos = None
        self.prev_obs_pos = None
        
        # Previous rotation data
        self.prev_ego_rot = None
        self.prev_obs_rot = None
        
        if self.debug:
            logger.info("RelativePositionTracker initialized with dt =", dt)
        
    def update(self, observation, tick):
        """
        Update the tracker with current vehicle positions from observation
        
        Args:
            observation: Tuple containing (ego_pos, obstacle_pos, ego_rot, obstacle_rot, tick)
            tick: Current simulation tick
        """
        try:
            if observation is None:
                if self.debug:
                    logger.info(f"Warning: Null observation at tick {tick}")
                return
                
            # Unpack observation - assuming new format with rotation
            if len(observation) < 5:
                if self.debug:
                    logger.info(f"Error: Invalid observation format at tick {tick}. Expected 5 elements, got {len(observation)}")
                return
                
            ego_pos, obstacle_pos, ego_rot, obstacle_rot, _ = observation
            
            # Validate position data
            if ego_pos is None or obstacle_pos is None:
                if self.debug:
                    logger.info(f"Error: Missing position data at tick {tick}")
                return
                
            # Validate rotation data
            if ego_rot is None or obstacle_rot is None:
                if self.debug:
                    logger.info(f"Error: Missing rotation data at tick {tick}")
                return
            
            # Convert yaw from degrees to radians
            ego_yaw_rad = math.radians(ego_rot[0])
            obstacle_yaw_rad = math.radians(obstacle_rot[0])
            
            # Calculate relative yaw in radians
            rel_yaw = obstacle_yaw_rad - ego_yaw_rad
            
            # Calculate relative position between vehicles
            rel_x = obstacle_pos[0] - ego_pos[0]
            rel_y = obstacle_pos[1] - ego_pos[1]
            
            # Calculate velocities if we have previous positions
            rel_vx, rel_vy, rel_yaw_rate = 0.0, 0.0, 0.0
            
            if self.prev_ego_pos is not None and self.prev_obs_pos is not None:
                # Calculate ego vehicle velocity
                ego_vx = (ego_pos[0] - self.prev_ego_pos[0]) / self.dt
                ego_vy = (ego_pos[1] - self.prev_ego_pos[1]) / self.dt
                
                # Calculate obstacle vehicle velocity
                obs_vx = (obstacle_pos[0] - self.prev_obs_pos[0]) / self.dt
                obs_vy = (obstacle_pos[1] - self.prev_obs_pos[1]) / self.dt
                
                # Calculate relative velocity
                rel_vx = obs_vx - ego_vx
                rel_vy = obs_vy - ego_vy
                
                # Calculate yaw rate with rotation data
                if self.prev_ego_rot is not None and self.prev_obs_rot is not None:
                    # Get previous rotation values
                    prev_ego_yaw_rad = math.radians(self.prev_ego_rot[0])
                    prev_obs_yaw_rad = math.radians(self.prev_obs_rot[0])
                    
                    # Calculate yaw rates
                    ego_yaw_rate = (ego_yaw_rad - prev_ego_yaw_rad) / self.dt
                    obs_yaw_rate = (obstacle_yaw_rad - prev_obs_yaw_rad) / self.dt
                    
                    # Calculate relative yaw rate
                    rel_yaw_rate = obs_yaw_rate - ego_yaw_rate
            
            # Store current positions for next velocity calculation
            self.prev_ego_pos = ego_pos.copy()
            self.prev_obs_pos = obstacle_pos.copy()
            self.prev_ego_rot = ego_rot.copy()
            self.prev_obs_rot = obstacle_rot.copy()
            
            # Set initial state if first update
            if tick == self.last_update_time + 1:
                # Predict
                self.ekf.predict()
                
                # Update
                z = np.array([rel_x, rel_y, rel_yaw])
                self.ekf.update(z)
                
                if self.debug:
                    logger.info(f"Tick {tick}: Updated EKF with measurements [{rel_x:.2f}, {rel_y:.2f}, {rel_yaw:.2f}]")
            else:
                # Initialize state
                self.ekf.x = np.array([rel_x, rel_y, rel_yaw, rel_vx, rel_vy, rel_yaw_rate])
                
                if self.debug:
                    logger.info(f"Tick {tick}: Initialized EKF state with [{rel_x:.2f}, {rel_y:.2f}, {rel_yaw:.2f}, {rel_vx:.2f}, {rel_vy:.2f}, {rel_yaw_rate:.2f}]")
            
            self.last_update_time = tick
            
        except Exception as e:
            logger.info(f"Error updating tracker at tick {tick}: {str(e)}")
        
    def predict_future_position(self, steps):
        """
        Predict future relative positions
        
        Args:
            steps (int): Number of future steps to predict
            
        Returns:
            list: List of (x, y, yaw) tuples for each future step
        """
        try:
            # Current state
            state = self.ekf.x.copy()
            
            # Transition matrix for simulation
            F = np.array([
                [1, 0, 0, self.dt, 0, 0],
                [0, 1, 0, 0, self.dt, 0],
                [0, 0, 1, 0, 0, self.dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
            predictions = []
            for i in range(steps):
                # Apply transition
                state = F @ state
                
                # Extract position
                predictions.append((state[0], state[1], state[2]))
                
            return predictions
        except Exception as e:
            logger.info(f"Error predicting future positions: {str(e)}")
            return [(0.0, 0.0, 0.0)] * steps
        
    def get_position_uncertainty(self):
        """
        Get position uncertainty from covariance matrix
        
        Returns:
            numpy.ndarray: 2x2 position covariance matrix
        """
        try:
            # Extract position covariance
            pos_cov = self.ekf.P[:2, :2]
            return pos_cov
        except Exception as e:
            logger.info(f"Error getting position uncertainty: {str(e)}")
            return np.eye(2)  # Return identity matrix as fallback


def calculate_collision_probability_relative(rel_tracker, predicted_positions, ego_radius=2.0, obstacle_radius=2.0):
    """
    Calculate collision probability based on relative positions and uncertainties
    
    Args:
        rel_tracker (RelativePositionTracker): Tracker containing position uncertainty information
        predicted_positions (list): List of (x, y, yaw) tuples for predicted relative positions
        ego_radius (float): Approximate radius of the ego vehicle (meters)
        obstacle_radius (float): Approximate radius of the obstacle vehicle (meters)
        
    Returns:
        tuple: (max_probability, collision_time, list_of_probabilities)
    """
    collision_probabilities = []
    collision_times = []
    
    # Combined radius for collision detection
    combined_radius = ego_radius + obstacle_radius
    
    for i, pos in enumerate(predicted_positions):
        rel_x, rel_y, _ = pos
        
        # Distance between vehicles
        distance = math.sqrt(rel_x**2 + rel_y**2)
        
        # Get uncertainty
        pos_cov = rel_tracker.get_position_uncertainty()
        
        # Simplified collision probability calculation
        # Assuming normal distribution
        if distance < combined_radius:
            # Already colliding
            prob = 1.0
        else:
            # Calculate probability using Mahalanobis distance
            d_squared = np.array([rel_x, rel_y]) @ np.linalg.inv(pos_cov) @ np.array([rel_x, rel_y])
            prob = 1 - math.exp(-0.5 * d_squared) 
            
            # Scale based on distance to collision boundary
            scale_factor = combined_radius / max(distance, combined_radius)
            prob *= scale_factor
        
        collision_probabilities.append(prob)
        
        if prob > 0.5:  # Arbitrary threshold for reporting collision time
            collision_times.append(i)
    
    # Return max probability and earliest collision time
    max_prob = max(collision_probabilities) if collision_probabilities else 0.0
    collision_time = min(collision_times) if collision_times else -1
    
    return max_prob, collision_time, collision_probabilities



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
            action: Control input for the ego vehicle. Can be:
                   - None: Will not apply any control
                   - carla.VehicleControl: Direct control to apply
                   - list/tuple: [throttle, steer, brake(optional)] values
        
        Returns:
            observation: Current observation
        """
        # Apply ego vehicle control if provided
        if action is not None:
            if isinstance(action, carla.VehicleControl):
                # Apply directly if it's already a VehicleControl object
                self.ego_vehicle.apply_control(action)
            elif isinstance(action, (list, tuple)) and len(action) >= 2:
                # Convert list/tuple to VehicleControl
                ego_control = carla.VehicleControl(
                    throttle=float(action[0]), 
                    steer=float(action[1]),
                    brake=float(action[2]) if len(action) > 2 else 0.0
                )
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
                        default='merge',
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
        source_rate=1000000000000.0,  # 10 Mbps
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
    rel_tracker = RelativePositionTracker(
        dt=base_config['simulation']['delta_seconds']
    )
    
    # Initialize co-simulator
    co_sim = CarlaCoSimulator(network_sim, simulator, timestep=0.01)

    brake = False
    
    logger.info(f"Starting simulation with {base_config['simulation']['prediction_steps']} prediction steps")
    logger.info(f"Emergency brake threshold: {base_config['simulation']['emergency_brake_threshold']}")

    # Run for specified number of steps
    for cur_step in range(base_config['ego_vehicle']['go_straight_ticks'] + base_config['ego_vehicle']['turn_ticks'] + base_config['ego_vehicle']['after_turn_ticks']):
        
        # Generate the appropriate control for the ego vehicle
        ego_control = None
        
        if brake:
            # Emergency brake
            ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        else:
            # Normal driving based on config
            control = carla.VehicleControl()
            
            if cur_step < base_config['ego_vehicle']['go_straight_ticks']:
                control.throttle = base_config['ego_vehicle']['throttle']['straight']
                control.steer = base_config['ego_vehicle']['steer']['straight'] if 'straight' in base_config['ego_vehicle']['steer'] else 0.0
            elif cur_step < base_config['ego_vehicle']['go_straight_ticks'] + base_config['ego_vehicle']['turn_ticks']:
                control.throttle = base_config['ego_vehicle']['throttle']['turn']
                control.steer = base_config['ego_vehicle']['steer']['turn']
            else:
                control.throttle = base_config['ego_vehicle']['throttle']['after_turn']
                control.steer = base_config['ego_vehicle']['steer']['after_turn'] if 'after_turn' in base_config['ego_vehicle']['steer'] else 0.0
            
            ego_control = control
        
        # Step the simulation with calculated ego control
        observation = co_sim.step(ego_control)
        
        # Reset brake flag
        brake = False
        
        # Predict future positions and calculate collision probability
        if observation is not None:
            # Update the tracker with the observation data
            rel_tracker.update(observation, cur_step)
            
            # Predict future positions
            predicted_positions = rel_tracker.predict_future_position(
                int(base_config['simulation']['prediction_steps']))
            
            # Calculate collision probabilities
            max_collision_prob, collision_time, collision_probabilities = calculate_collision_probability_relative(
                rel_tracker, predicted_positions)
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
