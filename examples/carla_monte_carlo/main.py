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
        self.rel_tracker = None
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
        self.tick = 0
        self.obstacle_buffer = []
        
        # Spawn the vehicles and sensors
        self._spawn_actors()
        
        # Initialize tracker
        self.rel_tracker = RelativePositionTracker(
            self.ego_vehicle,
            self.obstacle_vehicle,
            dt=self.config['simulation']['delta_seconds'])
            
        # Tick the world to get initial observation
        self.world.tick()
        
        # Return the initial observation
        return self._get_observation()
    
    def step(self, action=None):
        """
        Take a step in the simulation.
        
        Returns:
            tuple: (observation, reward, done, info)
        """
        if self.has_collided:
            return self._get_observation(), -1000, True, {"collision": True}
            
        # Increment tick counter
        self.tick += 1
        
        # Store current observation to buffer
        current_observation = (self.ego_vehicle.get_transform(), 
                              self.obstacle_vehicle.get_transform(), 
                              self.tick)
        self.obstacle_buffer.append(current_observation)
        
        # Update tracker with real-time data
        self.rel_tracker.update(self.tick)
        
        brake = False
        max_collision_prob = 0.0
    
        # Predict future positions using EKF
        predicted_positions = self.rel_tracker.predict_future_position(
            int(self.config['simulation']['prediction_steps']))

        # Calculate collision probabilities with new relative position approach
        max_collision_prob, collision_time, collision_probabilities = calculate_collision_probability_relative(
            self.rel_tracker, predicted_positions)
        
        # Apply emergency brake if collision probability exceeds threshold
        if max_collision_prob > self.config['simulation']['emergency_brake_threshold']:
            # Emergency brake
            brake = True
    
        # Apply vehicle controls based on action or default behavior
        self._apply_vehicle_controls(brake, action=action)
        
        # Tick the world
        self.world.tick()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward (negative for collisions, slight penalty for braking, positive for moving)
        reward = 0
        if self.has_collided:
            reward = -1000  # Large negative reward for collision
        elif brake:
            reward = -10    # Slight penalty for emergency braking
        else:
            reward = 1      # Small positive reward for moving
            
        # Check if done
        done = self.has_collided or self.tick >= (self.config['ego_vehicle']['go_straight_ticks'] + 
                                                 self.config['ego_vehicle']['turn_ticks'] + 
                                                 self.config['ego_vehicle']['after_turn_ticks'])
                                                 
        # Create info dict
        info = {
            "collision": self.has_collided,
            "collision_probability": max_collision_prob,
            "tick": self.tick
        }
        
        return observation, reward, done, info
    
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
        # For this implementation, we use camera frames as observations
        if self.frame_queue:
            return self.frame_queue[-1]
        return None
    
    def _apply_vehicle_controls(self, brake, action=None):
        """
        Apply controls to vehicles based on brake flag or custom action.
        
        Args:
            brake (bool): Whether to apply emergency brake
            action (ndarray, optional): Custom action to apply instead of default behavior
        """
        # For ego vehicle
        if brake:
            # Emergency brake
            self.ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        elif action is not None:
            # Apply custom action if provided
            # Assuming action format is [throttle, steer]
            if len(action) >= 2:
                ego_control = carla.VehicleControl(
                    throttle=float(action[0]), 
                    steer=float(action[1]),
                    brake=float(action[2]) if len(action) > 2 else 0.0
                )
                self.ego_vehicle.apply_control(ego_control)
        else:
            # Normal driving based on config
            ego_control = carla.VehicleControl()
            if self.tick < self.config['ego_vehicle']['go_straight_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['straight']
            elif self.tick < self.config['ego_vehicle']['go_straight_ticks'] + self.config[
                    'ego_vehicle']['turn_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['turn']
                ego_control.steer = self.config['ego_vehicle']['steer']['turn']
            else:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['after_turn']
                
            self.ego_vehicle.apply_control(ego_control)

        # For obstacle vehicle (follows predetermined path)
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


def run_adaptive_simulation(config, output_dir):
    """Run a simulation with braking based on collision probability using the new interface"""
    simulator = CarlaLatencySimulator(config, output_dir)
    has_collided = False
    
    try:
        # Reset to initialize the simulation
        simulator.reset()
        
        # Run simulation loop
        done = False
        while not done:
            # Step simulation with default behavior
            _, _, done, info = simulator.step()
            has_collided = info.get("collision", False)
            
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Clean up
        simulator.close()
        return has_collided

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
        source_rate=1.0,  # 10 Mbps
        weights=[1, 2],       # Weight client->server flows lower than server->client
        debug=True
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
    simulator.reset()

    co_sim = CarlaCoSimulator(network_sim, simulator, timestep=0.01)

    try:
        # Run for specified number of steps
        for step in range(base_config['ego_vehicle']['go_straight_ticks'] + base_config['ego_vehicle']['turn_ticks'] + base_config['ego_vehicle']['after_turn_ticks']):
            
            # Step the simulation
            observation, reward, done, info = co_sim.step(None)
            
            print(f"Step {step}: reward={reward}, done={done}")
            print(f"  - Collision probability: {info.get('collision_probability', 0):.4f}")
            
            if done:
                print(f"Simulation ended after {step} steps")
                break
                
        print("Simulation completed")
        
    except KeyboardInterrupt:
        print("Simulation interrupted by user")
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        simulator.close()

if __name__ == '__main__':
    main()
