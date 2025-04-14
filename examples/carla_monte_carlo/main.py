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


def run_adaptive_simulation(config, output_dir):
    """Run a simulation with adaptive delta_k and braking based on collision probability"""
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'],
                          config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()

    # Setup ego vehicle spawn point
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config[
        'obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config[
        'obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config[
        'obstacle_vehicle']['spawn_offset']['yaw']

    # Spawn both vehicles
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)

    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    # Attach camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(config['video']['width']))
    camera_bp.set_attribute('image_size_y', str(config['video']['height']))
    camera_bp.set_attribute('fov', config['camera']['fov'])

    camera_transform = carla.Transform(
        carla.Location(
            x=ego_spawn_point.location.x + config['camera']['offset']['x'],
            y=ego_spawn_point.location.y + config['camera']['offset']['y'],
            z=config['camera']['height']), carla.Rotation(pitch=-90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=None)

    frame_queue = []

    def camera_callback(image):
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()
        frame_queue.append(frame_bgr)

    camera.listen(camera_callback)

    # Initialize our new relative position tracker
    rel_tracker = RelativePositionTracker(
        ego_vehicle,
        obstacle_vehicle,
        dt=config['simulation']['delta_seconds'])

    # Initialize tracker with initial delta_k
    initial_delta_k = config['simulation']['delta_k']
    current_delta_k = initial_delta_k

    # Initialize obstacle buffer for delayed observations
    obstacle_buffer = []

    # Add collision sensor
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp,
                                         carla.Transform(),
                                         attach_to=ego_vehicle)

    has_collided = False

    def collision_callback(event):
        nonlocal has_collided
        has_collided = True

    collision_sensor.listen(collision_callback)

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] +
                          config['ego_vehicle']['turn_ticks'] +
                          config['ego_vehicle']['after_turn_ticks']):

            world.tick()

            if has_collided:
                ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0))
                obstacle_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0))
                break

            # Store current observation
            current_observation = (ego_vehicle.get_transform(), obstacle_vehicle.get_transform(), tick)
            obstacle_buffer.append(current_observation)
            
            # Update tracker with real-time data initially
            rel_tracker.update(tick)
            
            brake = False
            max_collision_prob = 0.0

            if tick >= config['simulation']['l_max']:
                # Use delayed observation based on delta_k
                obstacle_buffer.pop(0)
                historical_observation = obstacle_buffer[0]
                
                # Predict future positions using EKF
                predicted_positions = rel_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] / current_delta_k))

                # Calculate collision probabilities with new relative position approach
                max_collision_prob, collision_time, collision_probabilities = calculate_collision_probability_relative(
                    rel_tracker, predicted_positions)
                print(f"Collision probability: {max_collision_prob:.4f}")

                # Adaptive behavior based on collision probability
                if max_collision_prob > config['simulation'][
                        'emergency_brake_threshold']:
                    # Emergency brake
                    brake = True
                    print(
                        f"Emergency brake activated! Collision probability: {max_collision_prob:.4f}"
                    )

                elif max_collision_prob > config['simulation'][
                        'cautious_threshold']:
                    # Increase tracking frequency (decrease delta_k)
                    new_delta_k = config['simulation']['cautious_delta_k']
                    if new_delta_k != current_delta_k:
                        print(
                            f"Adjusting delta_k from {current_delta_k} to {new_delta_k}"
                        )
                        current_delta_k = new_delta_k
                        # Drop observations from buffer to match new delta_k
                        for i in range(config['simulation']['l_max'] - new_delta_k):
                            if obstacle_buffer:
                                obstacle_buffer.pop(0)

            if brake:
                ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
            else:
                # Normal driving
                ego_control = carla.VehicleControl()
                if tick < config['ego_vehicle']['go_straight_ticks']:
                    ego_control.throttle = config['ego_vehicle']['throttle'][
                        'straight']
                elif tick < config['ego_vehicle']['go_straight_ticks'] + config[
                        'ego_vehicle']['turn_ticks']:
                    ego_control.throttle = config['ego_vehicle']['throttle'][
                        'turn']
                    ego_control.steer = config['ego_vehicle']['steer']['turn']
                else:
                    ego_control.throttle = config['ego_vehicle']['throttle'][
                        'after_turn']

            ego_vehicle.apply_control(ego_control)

            # Apply controls
            obstacle_control = carla.VehicleControl()

            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                # Initial straight phase
                obstacle_control.throttle = config['obstacle_vehicle'][
                    'throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer'][
                    'straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config[
                    'obstacle_vehicle']['turn_ticks']:
                # Turning phase
                obstacle_control.throttle = config['obstacle_vehicle'][
                    'throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer'][
                    'turn']
            else:
                # After turn straight phase
                obstacle_control.throttle = config['obstacle_vehicle'][
                    'throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer'][
                    'after_turn']

            obstacle_vehicle.apply_control(obstacle_control)

            # Process camera frames
            while frame_queue:
                frame_queue.pop(0)

            timestamp = tick * config['simulation']['delta_seconds']

    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Cleanup
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()

        camera.stop()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        if camera is not None:
            camera.destroy()

        client.reload_world()
        world.apply_settings(original_settings)
        return has_collided, current_delta_k

def get_monte_carlo_spawn_point(config, ego_spawn_point, std_dev=0):
    """
    Generate a randomized spawn point for the obstacle vehicle using Monte Carlo sampling.
    
    Args:
        config: Configuration dictionary
        ego_spawn_point: Base ego vehicle spawn point
        std_dev: Standard deviation for the normal distribution (in meters)
    
    Returns:
        carla.Transform: Randomized spawn point
    """
    base_x = ego_spawn_point.location.x + config['obstacle_vehicle'][
        'spawn_offset']['x']
    base_y = ego_spawn_point.location.y + config['obstacle_vehicle'][
        'spawn_offset']['y']
    base_yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle'][
        'spawn_offset']['yaw']

    # Sample from normal distribution for position and yaw
    x = np.random.normal(base_x, std_dev)
    y = np.random.normal(base_y, std_dev)
    yaw = np.random.normal(base_yaw,
                           std_dev * 2)  # Larger variation in orientation

    spawn_point = carla.Transform(carla.Location(x=x, y=y, z=0.0),
                                  carla.Rotation(yaw=yaw))

    return spawn_point


def run_monte_carlo_simulation(config, num_samples=10, output_dir='./results'):
    """
    Run multiple simulations with Monte Carlo sampling of obstacle spawn points.
    
    Args:
        config: Configuration dictionary
        num_samples: Number of Monte Carlo samples to run
    
    Returns:
        dict: Statistics about collisions and spawn points
    """
    collision_stats = {
        'num_collisions': 0,
        'spawn_points': [],
        'collision_cases': []
    }

    # Connect to CARLA
    client = carla.Client(config['simulation']['host'],
                          config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")

    # Get base spawn point for ego vehicle
    spawn_points = world.get_map().get_spawn_points()
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']

    for sample in range(num_samples):
        print(f"\nRunning Monte Carlo sample {sample + 1}/{num_samples}")

        # Generate random spawn point
        obstacle_spawn_point = get_monte_carlo_spawn_point(
            config, ego_spawn_point)

        # Update config with new spawn point
        sample_config = dict(config)
        sample_config['obstacle_vehicle']['spawn_offset'] = {
            'x': obstacle_spawn_point.location.x - ego_spawn_point.location.x,
            'y': obstacle_spawn_point.location.y - ego_spawn_point.location.y,
            'yaw': obstacle_spawn_point.rotation.yaw -
                   ego_spawn_point.rotation.yaw
        }

        try:
            # Run simulation and check for collision
            has_collided, current_delta_k = run_adaptive_simulation(
                sample_config, output_dir)

            # Check if collision occurred
            if has_collided:
                collision_stats['num_collisions'] += 1
                collision_stats['collision_cases'].append({
                    'spawn_point': obstacle_spawn_point,
                    'sample_num': sample
                })
            else:
                collision_stats['spawn_points'].append(obstacle_spawn_point)

        except Exception as e:
            print(f"Error in sample {sample}: {e}")
            collision_stats['num_collisions'] += 1
            collision_stats['collision_cases'].append({
                'spawn_point': obstacle_spawn_point,
                'sample_num': sample,
                'error': str(e)
            })

    return collision_stats


def main():
    parser = argparse.ArgumentParser(
        description='Run CARLA simulation with configurable parameters')
    parser.add_argument('--cautious_delta_k',
                        type=int,
                        default=-1,
                        help='Value for cautious_delta_k parameter')
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

    # Select configuration based on argument
    config_map = {
        'right_turn': unprotected_right_turn_config,
        'left_turn': unprotected_left_turn_config,
        'merge': opposite_direction_merge_config
    }

    base_config = config_map[args.config_type]

    # Update configuration with command line parameters
    if args.cautious_delta_k != -1:
        base_config['simulation']['cautious_delta_k'] = args.cautious_delta_k
        base_config['simulation']['l_max'] = args.cautious_delta_k
        base_config['simulation']['delta_k'] = args.cautious_delta_k

    # Update emergency brake threshold
    base_config['simulation'][
        'emergency_brake_threshold'] = args.emergency_brake_threshold

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    max_retries = 3
    retry_count = 0

    # Base configuration for Monte Carlo simulation
    num_samples = 1

    while retry_count < max_retries:
        try:
            # Run Monte Carlo simulation
            stats = run_monte_carlo_simulation(base_config, num_samples,
                                               args.output_dir)

            # Calculate statistics
            collision_rate = stats['num_collisions'] / num_samples

            print(f"\nMonte Carlo simulation completed.")
            print(
                f"Collision rate: {collision_rate:.2%} ({stats['num_collisions']}/{num_samples} collisions)"
            )
            break

        except Exception as e:
            retry_count += 1
            print(f"\nError occurred: {e}")
            print(f"Retry {retry_count}/{max_retries}")

            print(
                "CARLA connection issue detected. Attempting to restart CARLA..."
            )
            if restart_carla_docker():
                print("CARLA successfully restarted")
                time.sleep(5)  # Give additional time for CARLA to stabilize
            else:
                print("Failed to restart CARLA")

            if retry_count >= max_retries:
                print("Max retries reached. Exiting...")
                break

            time.sleep(5)  # Wait before retrying


if __name__ == '__main__':
    main()
