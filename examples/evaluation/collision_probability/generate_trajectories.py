#!/usr/bin/env python3
"""
Generate real CARLA trajectory data for network timing comparison
"""

import os
import sys
import time
import csv
import carla
from main import unprotected_right_turn_config


def generate_real_trajectories(config, ego_file='ego_trajectory_real.csv', obstacle_file='obstacle_trajectory_real.csv'):
    """Generate actual trajectories from CARLA simulation"""
    
    # Clear existing files
    if os.path.exists(ego_file):
        os.remove(ego_file)
    if os.path.exists(obstacle_file):
        os.remove(obstacle_file)
    
    # Connect to CARLA
    client = carla.Client(config['simulation']['host'], config['simulation']['port'])
    client.set_timeout(10.0)
    world = client.load_world("Town03")
    
    # Set synchronous mode
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.fixed_delta_seconds = config['simulation']['delta_seconds']
    settings.synchronous_mode = True
    settings.no_rendering_mode = True  # Faster without rendering
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    
    # Setup spawn points
    ego_spawn_point = spawn_points[0]
    ego_spawn_point.location.x += config['ego_vehicle']['spawn_offset']['x']
    ego_spawn_point.location.y += config['ego_vehicle']['spawn_offset']['y']
    ego_spawn_point.rotation.yaw += config['ego_vehicle']['spawn_offset']['yaw']
    
    obstacle_spawn_point = spawn_points[1]
    obstacle_spawn_point.location.x = ego_spawn_point.location.x + config['obstacle_vehicle']['spawn_offset']['x']
    obstacle_spawn_point.location.y = ego_spawn_point.location.y + config['obstacle_vehicle']['spawn_offset']['y']
    obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + config['obstacle_vehicle']['spawn_offset']['yaw']
    
    # Spawn vehicles
    ego_bp = blueprint_library.find(config['ego_vehicle']['model'])
    ego_bp.set_attribute('role_name', 'ego')
    ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn_point)
    
    obstacle_bp = blueprint_library.find(config['obstacle_vehicle']['model'])
    obstacle_bp.set_attribute('role_name', 'obstacle')
    obstacle_vehicle = world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)
    
    if not ego_vehicle or not obstacle_vehicle:
        print("Failed to spawn vehicles")
        return False
    
    # Set autopilot off
    ego_vehicle.set_autopilot(False)
    obstacle_vehicle.set_autopilot(False)
    
    try:
        total_ticks = (config['ego_vehicle']['go_straight_ticks'] + 
                      config['ego_vehicle']['turn_ticks'] + 
                      config['ego_vehicle']['after_turn_ticks'])
        
        ego_trajectories = []
        obstacle_trajectories = []
        
        print(f"Generating trajectories for {total_ticks} ticks...")
        
        for tick in range(total_ticks):
            world.tick()
            
            # Control ego vehicle
            ego_control = carla.VehicleControl()
            if tick < config['ego_vehicle']['go_straight_ticks']:
                ego_control.throttle = config['ego_vehicle']['throttle']['straight']
                ego_control.steer = 0.0
            elif tick < config['ego_vehicle']['go_straight_ticks'] + config['ego_vehicle']['turn_ticks']:
                ego_control.throttle = config['ego_vehicle']['throttle']['turn']
                ego_control.steer = config['ego_vehicle']['steer']['turn']
            else:
                ego_control.throttle = config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = 0.0
            
            ego_vehicle.apply_control(ego_control)
            
            # Control obstacle vehicle
            obstacle_control = carla.VehicleControl()
            if tick < config['obstacle_vehicle']['go_straight_ticks']:
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['straight']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['straight']
            elif tick < config['obstacle_vehicle']['go_straight_ticks'] + config['obstacle_vehicle']['turn_ticks']:
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['turn']
            else:
                obstacle_control.throttle = config['obstacle_vehicle']['throttle']['after_turn']
                obstacle_control.steer = config['obstacle_vehicle']['steer']['after_turn']
            
            obstacle_vehicle.apply_control(obstacle_control)
            
            # Record positions
            ego_transform = ego_vehicle.get_transform()
            obstacle_transform = obstacle_vehicle.get_transform()
            
            ego_trajectories.append([
                ego_transform.location.x,
                ego_transform.location.y,
                ego_transform.rotation.yaw
            ])
            
            obstacle_trajectories.append([
                obstacle_transform.location.x,
                obstacle_transform.location.y,
                obstacle_transform.rotation.yaw
            ])
            
            if tick % 100 == 0:
                print(f"Tick {tick}/{total_ticks}")
        
        # Save trajectories
        with open(ego_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for traj in ego_trajectories:
                writer.writerow(traj)
        
        with open(obstacle_file, 'w', newline='') as f:
            writer = csv.writer(f)
            for traj in obstacle_trajectories:
                writer.writerow(traj)
        
        print(f"Saved {len(ego_trajectories)} ego trajectory points to {ego_file}")
        print(f"Saved {len(obstacle_trajectories)} obstacle trajectory points to {obstacle_file}")
        
        # Show sample data
        print(f"\nSample ego positions:")
        for i in range(min(5, len(ego_trajectories))):
            print(f"  {i}: x={ego_trajectories[i][0]:.2f}, y={ego_trajectories[i][1]:.2f}, yaw={ego_trajectories[i][2]:.2f}")
        
        print(f"\nSample obstacle positions:")
        for i in range(min(5, len(obstacle_trajectories))):
            print(f"  {i}: x={obstacle_trajectories[i][0]:.2f}, y={obstacle_trajectories[i][1]:.2f}, yaw={obstacle_trajectories[i][2]:.2f}")
        
        return True
        
    finally:
        if ego_vehicle:
            ego_vehicle.destroy()
        if obstacle_vehicle:
            obstacle_vehicle.destroy()
        
        world.apply_settings(original_settings)
        client.reload_world()


if __name__ == "__main__":
    config = unprotected_right_turn_config.copy()
    generate_real_trajectories(config)