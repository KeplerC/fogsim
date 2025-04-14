from abc import ABC
from abc import abstractmethod
import argparse
import math
import os
import shutil
import time

import carla
import cv2
from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.stats import norm

from configs import *
from utils import *


def save_trajectory(vehicle, filename):
    """Save vehicle transform to file"""
    with open(filename, 'a') as f:
        transform = vehicle.get_transform()
        f.write(
            f"{transform.location.x},{transform.location.y},{transform.rotation.yaw}\n"
        )


def load_trajectory(filename):
    """Load trajectory from file"""
    trajectory = []
    with open(filename, 'r') as f:
        for line in f:
            x, y, yaw = map(float, line.strip().split(','))
            trajectory.append([x, y, math.radians(yaw)])
    return trajectory


def run_first_simulation(config, trajectory_file=None):
    """Run the first simulation to generate the ego vehicle trajectory"""
    # Use trajectory file from config if none provided
    if trajectory_file is None:
        trajectory_file = config['trajectories']['ego']

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

    # Spawn only the ego vehicle
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    ego_vehicle.set_autopilot(False)

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] +
                          config['ego_vehicle']['turn_ticks'] +
                          config['ego_vehicle']['after_turn_ticks']):
            world.tick()

            # Apply controls and save trajectory
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                # Initial straight phase
                ego_control.throttle = config['ego_vehicle']['throttle'][
                    'straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config[
                    'ego_vehicle']['turn_ticks']:
                # Turning phase
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                ego_control.throttle = config['ego_vehicle']['throttle'][
                    'after_turn']
                ego_control.steer = 0.0

            ego_vehicle.apply_control(ego_control)
            save_trajectory(ego_vehicle, trajectory_file)

    finally:
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        client.reload_world()
        world.apply_settings(original_settings)


def run_simulation(config, output_dir):
    """Run a simulation with the given configuration using pre-recorded trajectory"""
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

    # Find two spawn points close to each other
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

    # Spawn the ego vehicle
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')

    # Spawn both vehicles
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    # Load the reference trajectory
    ego_trajectory = load_trajectory(config['trajectories']['ego'])
    tracker_type = "ekf"  # Options: "kf", "ekf", "ground_truth"
    if tracker_type == "ground_truth":
        obstacle_tracker = GroundTruthTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])
    elif tracker_type == "ekf":
        obstacle_tracker = EKFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])
    else:  # "kf"
        obstacle_tracker = KFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])

    # No autopilot, we will manually control both
    ego_vehicle.set_autopilot(False)
    if obstacle_vehicle is not None:
        obstacle_vehicle.set_autopilot(False)

    # Attach a top-down BEV camera above the intersection or ego vehicle's start
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

    # Only create video writer if save_video is True
    video_writer = None
    if config['save_options']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            config['video']['filename'], fourcc, config['video']['fps'],
            (config['video']['width'], config['video']['height']))

    collision_prob = 0.0
    frame_queue = []

    def camera_callback(image):
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()

        frame_queue.append(frame_bgr)

        # Only save individual frames if save_images is True
        if config['save_options']['save_images']:
            cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

    camera.listen(camera_callback)

    # Initialize obstacle tracker with dt from simulation settings
    obstacle_buffer = [obstacle_vehicle.get_transform()
                      ] * config['simulation']['l_max']

    # Add collision sensor
    collision_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(collision_bp,
                                         carla.Transform(),
                                         attach_to=ego_vehicle)

    # Collision flag
    has_collided = False

    def collision_callback(event):
        nonlocal has_collided
        has_collided = True
        # print(f"Collision detected with {event.other_actor}")

    collision_sensor.listen(collision_callback)

    # Add CSV setup for collision probability logging
    collision_prob_file = os.path.join(output_dir,
                                       'collision_probabilities.csv')

    if os.path.exists(collision_prob_file):
        pass
    else:
        with open(os.path.join(output_dir, 'collision_probabilities.csv'),
                  'w') as f:
            f.write('timestamp,tick,delta_k,collision_probability\n')

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] +
                          config['ego_vehicle']['turn_ticks'] +
                          config['ego_vehicle']['after_turn_ticks']):
            world.tick()

            # Check for collision
            if has_collided:
                print("Collision detected! Stopping vehicles.")
                # Stop both vehicles
                ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0))
                if obstacle_vehicle is not None:
                    obstacle_vehicle.apply_control(
                        carla.VehicleControl(throttle=0.0, brake=1.0))
                break

            # Update buffer with current transform
            current_transform = obstacle_vehicle.get_transform()
            obstacle_buffer.append(current_transform)
            obstacle_buffer.pop(0)

            # Only update obstacle tracking at delta_k intervals
            # if tick % config['simulation']['delta_k'] == 0:
            # Get historical transform from l_max steps ago
            historical_transform = obstacle_buffer[
                0]  # oldest transform in buffer

            obstacle_tracker.update((historical_transform.location.x,
                                     historical_transform.location.y,
                                     historical_transform.rotation.yaw), tick)

            predicted_ego_positions = obstacle_tracker.predict_future_position(
                int(config['simulation']['prediction_steps'] /
                    config['simulation']['delta_k']))
            collision_probabilities = []

            for step, predicted_pos in enumerate(predicted_ego_positions):
                if tick + step < len(ego_trajectory):
                    ego_trajectory_point = ego_trajectory[tick + step]
                    predicted_pos = [
                        predicted_pos[0], predicted_pos[1], predicted_pos[2]
                    ]
                    collision_prob = obstacle_tracker.calculate_collision_probability_with_trajectory(
                        ego_trajectory_point, predicted_pos)
                    collision_probabilities.append(collision_prob)

            # print (predicted_ego_positions, collision_probabilities) mapping

            collision_prob = max(collision_probabilities)
            collision_time = collision_probabilities.index(collision_prob)

            # print(f"Tick {tick}: Max collision probability: {collision_prob:.4f} at time step {collision_time}")

            # Vehicle control logic remains the same
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                # Initial straight phase
                ego_control.throttle = config['ego_vehicle']['throttle'][
                    'straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config[
                    'ego_vehicle']['turn_ticks']:
                # Turning phase
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                # After turn straight phase
                ego_control.throttle = config['ego_vehicle']['throttle'][
                    'after_turn']
                ego_control.steer = 0.0

            ego_vehicle.apply_control(ego_control)

            obstacle_control = carla.VehicleControl()
            obstacle_control.throttle = config['obstacle_vehicle']['throttle']
            obstacle_control.steer = config['obstacle_vehicle']['steer']
            obstacle_vehicle.apply_control(obstacle_control)

            # Write queued frames to video
            while frame_queue:
                frame_bgr = frame_queue.pop(0)

                # Add collision probability text to the frame
                collision_text = f"Collision Probability: {collision_prob:.4f}"
                cv2.putText(frame_bgr, collision_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                if config['save_options']['save_video']:
                    video_writer.write(frame_bgr)
                if config['save_options']['save_images']:
                    cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

            # After calculating collision_prob and collision_time
            timestamp = tick * config['simulation']['delta_seconds']
            with open(collision_prob_file, 'a') as f:
                f.write(
                    f'{timestamp:.2f},{tick},{config["simulation"]["delta_k"]},{collision_prob:.4f}\n'
                )

    finally:
        # Add collision sensor cleanup
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()

        # Cleanup
        camera.stop()
        if video_writer is not None:
            video_writer.release()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        if camera is not None:
            camera.destroy()

        client.reload_world()
        world.apply_settings(original_settings)
        # print(f"Video saved to {config['video']['filename']}")


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

    # Video writer setup
    video_writer = None
    if config['save_options']['save_video']:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            config['video']['filename'], fourcc, config['video']['fps'],
            (config['video']['width'], config['video']['height']))

    frame_queue = []

    def camera_callback(image):
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3].copy()

        frame_queue.append(frame_bgr)

        if config['save_options']['save_images']:
            cv2.imwrite(f"./bev_images/frame_{tick}.png", frame_bgr)

    camera.listen(camera_callback)

    # Load trajectory and setup tracker
    ego_trajectory = load_trajectory(config['trajectories']['ego'])

    # Initialize tracker with initial delta_k
    initial_delta_k = config['simulation']['delta_k']
    current_delta_k = initial_delta_k

    if config['simulation']['tracker_type'] == 'ekf':
        obstacle_tracker = EKFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])
    else:
        obstacle_tracker = KFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])

    # Add ground truth tracker (using same type as regular tracker)
    if config['simulation']['tracker_type'] == 'ekf':
        ground_truth_tracker = EKFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])
    else:
        ground_truth_tracker = KFObstacleTracker(
            ego_vehicle,
            obstacle_vehicle,
            dt=config['simulation']['delta_seconds'])

    # Initialize obstacle buffer
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
        # print(f"Collision detected with {event.other_actor}")

    collision_sensor.listen(collision_callback)

    # Setup CSV logging
    # collision_probabilities_lmax_config_type.csv
    collision_prob_file = os.path.join(output_dir,
                                       f'collision_probabilities_{config["simulation"]["l_max"]}.csv')
    if not os.path.exists(collision_prob_file):
        with open(collision_prob_file, 'w') as f:
            f.write(
                'timestamp,tick,delta_k,collision_probability,ground_truth_probability\n'
            )

    try:
        for tick in range(config['ego_vehicle']['go_straight_ticks'] +
                          config['ego_vehicle']['turn_ticks'] +
                          config['ego_vehicle']['after_turn_ticks']):

            world.tick()

            if has_collided:
                # print("Collision detected! Stopping vehicles.")
                ego_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0))
                obstacle_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, brake=1.0))
                break

            current_transform = obstacle_vehicle.get_transform()
            ground_truth_tracker.update(
                (current_transform.location.x, current_transform.location.y,
                 current_transform.rotation.yaw), tick)

            obstacle_buffer.append(current_transform)
            brake = False
            max_collision_prob = 0.0
            ground_truth_collision_prob = 0.0

            if tick >= config['simulation']['l_max']:
                obstacle_buffer.pop(0)

                historical_transform = obstacle_buffer[0]

                obstacle_tracker.update((historical_transform.location.x,
                                         historical_transform.location.y,
                                         historical_transform.rotation.yaw),
                                        tick)

                # Predict future positions and calculate collision probabilities
                predicted_positions = obstacle_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] /
                        current_delta_k))

                max_collision_prob, collision_time, collision_probabilities = calculate_collision_probabilities(
                    obstacle_tracker, predicted_positions, ego_trajectory, tick)
                ground_truth_predictions = ground_truth_tracker.predict_future_position(
                    int(config['simulation']['prediction_steps'] /
                        current_delta_k))

                ground_truth_max_prob, ground_truth_collision_time, ground_truth_probabilities = calculate_collision_probabilities(
                    ground_truth_tracker, ground_truth_predictions,
                    ego_trajectory, tick)
                # Get current ego vehicle position
                ego_transform = ego_vehicle.get_transform()
                ego_pos = (ego_transform.location.x, ego_transform.location.y,
                           math.radians(ego_transform.rotation.yaw))
                obstacle_pos = (current_transform.location.x,
                                current_transform.location.y,
                                math.radians(current_transform.rotation.yaw))
                max_prob_idx = collision_probabilities.index(max_collision_prob)
                predicted_pos = predicted_positions[max_prob_idx]
                ego_predicted_pos = ego_trajectory[tick + max_prob_idx]
                ground_truth_collision_prob = ground_truth_max_prob
                # Get predicted position with highest collision probability
                # print(f"\nTick {tick}:")
                # print("obstacle vehicle current position:")
                # print(f"(x={obstacle_pos[0]:.2f}, y={obstacle_pos[1]:.2f}, yaw={obstacle_pos[2]:.2f})")
                # print(f"Predicted obstacle position with max collision prob {max_collision_prob:.4f}:")
                # print(f"(x={predicted_pos[0]:.2f}, y={predicted_pos[1]:.2f}, yaw={predicted_pos[2]:.2f})")
                # # ego vehicle predicted position
                # print(f"Current ego position: (x={ego_pos[0]:.2f}, y={ego_pos[1]:.2f}, yaw={ego_pos[2]:.2f})")

                # print(f"Ego vehicle predicted position:")
                # print(f"(x={ego_predicted_pos[0]:.2f}, y={ego_predicted_pos[1]:.2f}, yaw={ego_predicted_pos[2]:.2f})")
                # print(f"Groundtruth collision probability:{ground_truth_collision_prob:.4f}")

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
                        # drop the obstacle buffer to new delta_k
                        for i in range(config['simulation']['l_max'] -
                                       new_delta_k):
                            obstacle_pos = obstacle_buffer.pop(0)
                            # update obstacle tracker with the new position
                            obstacle_tracker.update((obstacle_pos.location.x,
                                                     obstacle_pos.location.y,
                                                     obstacle_pos.rotation.yaw),
                                                    tick)

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
                frame_bgr = frame_queue.pop(0)

                # Add collision probability text to the frame
                collision_text = f"Predicted Collision Probability: {max_collision_prob:.4f}"
                cv2.putText(frame_bgr, collision_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                # Add ground truth collision probability text to the frame
                ground_truth_collision_text = f"Groundtruth Collision Probability: {ground_truth_collision_prob:.4f}"
                cv2.putText(frame_bgr, ground_truth_collision_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                # Add delta_k text to the frame
                delta_k_text = f"Current Latency: {current_delta_k * 10} ms"
                cv2.putText(frame_bgr, delta_k_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

                if config['save_options']['save_video']:
                    video_writer.write(frame_bgr)
                if config['save_options']['save_images']:
                    cv2.imwrite(
                        os.path.join(
                            output_dir,
                            'bev_images/frame_{tick}.png').format(tick=tick),
                        frame_bgr)

            # Log data
            timestamp = tick * config['simulation']['delta_seconds']
            # Read existing content
            try:
                with open(collision_prob_file, 'r') as f:
                    existing_content = f.read()
            except FileNotFoundError:
                existing_content = ''

            # Write updated content
            with open(collision_prob_file, 'w') as f:
                f.write(
                    existing_content +
                    f'{timestamp:.2f},{tick},{current_delta_k},{max_collision_prob:.4f},{ground_truth_collision_prob:.4f}\n'
                )
    except Exception as e:
        print(f"Error in simulation: {e}")
    finally:
        # Cleanup
        if collision_sensor is not None:
            collision_sensor.stop()
            collision_sensor.destroy()

        camera.stop()
        if video_writer is not None:
            video_writer.release()

        if ego_vehicle is not None:
            ego_vehicle.destroy()
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()
        if camera is not None:
            camera.destroy()

        client.reload_world()
        world.apply_settings(original_settings)
        # print(f"Video saved to {config['video']['filename']}")
        return has_collided, current_delta_k


def run_obstacle_only_simulation(config, trajectory_file=None):
    """Run a simulation with only the obstacle vehicle to record its trajectory"""
    # Use trajectory file from config if none provided
    if trajectory_file is None:
        trajectory_file = config['trajectories']['obstacle']

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

    # Setup obstacle vehicle spawn point based on ego vehicle's position
    # (since obstacle spawn is defined relative to ego in config)
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

    # Spawn only the obstacle vehicle
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)

    if obstacle_vehicle is None:
        raise RuntimeError("Failed to spawn obstacle vehicle")

    try:
        # Run for the same duration as the full simulation
        total_ticks = (config['ego_vehicle']['go_straight_ticks'] +
                       config['ego_vehicle']['turn_ticks'] +
                       config['ego_vehicle'].get('after_turn_ticks', 0))

        for tick in range(total_ticks):
            world.tick()

            # Apply phase-based control to obstacle vehicle
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
            save_trajectory(obstacle_vehicle, trajectory_file)

    finally:
        if obstacle_vehicle is not None:
            obstacle_vehicle.destroy()

        # Restore original settings
        world.apply_settings(original_settings)
        print(f"Obstacle trajectory saved to {trajectory_file}")


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

        # Update trajectory files for this sample
        sample_config['trajectories'] = {
            'ego': f'./ego_trajectory_sample_{sample}.csv',
            'obstacle': f'./obstacle_trajectory_sample_{sample}.csv'
        }
        # try:
        #     # Generate obstacle trajectory
        #     run_obstacle_only_simulation(sample_config)
        # except Exception as e:
        #     # obstacle spawn point is invalid
        #     print(f"Error in sample {sample}: {e}")
        #     continue

        try:
            # Generate trajectories
            run_first_simulation(sample_config)

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

        # Clean up trajectory files
        for trajectory_file in sample_config['trajectories'].values():
            if os.path.exists(trajectory_file):
                os.remove(trajectory_file)

        if not os.path.exists(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv')):
            with open(
                    os.path.join(output_dir,
                                 'monte_carlo_results/statistics.csv'),
                    'w') as f:
                f.write(f"scenario,lmax,delta_k,collision,delta_k_used\n")
        # save and append the stats to a csv file
        # with open(os.path.join(output_dir, 'monte_carlo_results/statistics.csv'), 'a') as f:
        # doesn't support 'a', read and write instead
        with open(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv'),
                'r') as f:
            lines = f.readlines()
        with open(
                os.path.join(output_dir, 'monte_carlo_results/statistics.csv'),
                'w') as f:
            for line in lines:
                f.write(line)
            # scenario, lmax, delta_k, collision yes or no, delta_k_used
            scenario_type = config['video']['filename'].split('/')[-1].split(
                '_collision')[0]
            f.write(
                f"{scenario_type},{config['simulation']['l_max']},{current_delta_k},{has_collided},{config['simulation']['delta_k']}\n"
            )

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

    # Update output paths based on output directory
    base_config['video']['filename'] = os.path.join(args.output_dir,
                                                    'simulation.mp4')
    base_config['trajectories']['ego'] = os.path.join(args.output_dir,
                                                      'ego_trajectory.csv')
    base_config['trajectories']['obstacle'] = os.path.join(
        args.output_dir, 'obstacle_trajectory.csv')

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'bev_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'monte_carlo_results'),
                exist_ok=True)

    max_retries = 3
    retry_count = 0

    # Clean old files in the output directory
    if os.path.exists(os.path.join(args.output_dir, 'bev_images')):
        shutil.rmtree(os.path.join(args.output_dir, 'bev_images'))
    os.makedirs(os.path.join(args.output_dir, 'bev_images'))

    collision_prob_file = os.path.join(args.output_dir,
                                       'collision_probabilities.csv')
    if os.path.exists(collision_prob_file):
        os.remove(collision_prob_file)

    # Base configuration for Monte Carlo simulation
    num_samples = 1

    while retry_count < max_retries:
        try:
            # Run Monte Carlo simulation
            stats = run_monte_carlo_simulation(base_config, num_samples,
                                               args.output_dir)

            # Calculate and save statistics
            collision_rate = stats['num_collisions'] / num_samples

            # Save results
            stats_file = os.path.join(args.output_dir, 'monte_carlo_results',
                                      'statistics.txt')
            with open(stats_file, 'w') as f:
                f.write(f"Monte Carlo Simulation Results\n")
                f.write(f"Configuration:\n")
                f.write(f"  Config Type: {args.config_type}\n")
                f.write(
                    f"  Cautious Delta K: {base_config['simulation']['cautious_delta_k']}\n"
                )
                f.write(
                    f"  Emergency Brake Threshold: {base_config['simulation']['emergency_brake_threshold']}\n\n"
                )
                f.write(f"Results:\n")
                f.write(f"  Number of samples: {num_samples}\n")
                f.write(f"  Number of collisions: {stats['num_collisions']}\n")
                f.write(f"  Collision rate: {collision_rate:.2%}\n\n")

                f.write("Collision cases:\n")
                for case in stats['collision_cases']:
                    f.write(f"Sample {case['sample_num']}:\n")
                    f.write(
                        f"  Spawn point: x={case['spawn_point'].location.x:.2f}, "
                        f"y={case['spawn_point'].location.y:.2f}, "
                        f"yaw={case['spawn_point'].rotation.yaw:.2f}\n")
                    if 'error' in case:
                        f.write(f"  Error: {case['error']}\n")

            print(f"\nMonte Carlo simulation completed.")
            print(
                f"Collision rate: {collision_rate:.2%} ({stats['num_collisions']}/{num_samples} collisions)"
            )
            print(f"Results saved to {args.output_dir}")
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
