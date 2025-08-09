from experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spawn_traffic_cones,
    town04_spawn_walkers,
    update_walkers,
    obstacle_map_from_bbs,
    clear_obstacle_map,
    union_obstacle_map,
    mask_obstacle_map,
    DELTA_SECONDS
)

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from contextlib import contextmanager

# Configuration constants
SCENARIOS = [
    (20, [19, 21]),
    (21, [20, 22]),
    (22, [21, 23]),
]
NUM_RANDOM_CARS = 25
PERCEPTION_LATENCIES = [0]  # ms

# Simulation constants  
REPLAN_INTERVAL = 10  # Frames between replanning
DISTANCE_THRESHOLD = 10  # Distance to trigger replanning
VIDEO_FPS = 30

# Traffic cone and walker spawn positions
TRAFFIC_CONE_POSITIONS = [(284, -230), (287, -225)]
WALKER_POSITIONS = []

@contextmanager
def spawn_actors(world, destination_parking_spot, parked_spots):
    """Context manager for spawning and cleaning up actors."""
    actors_to_cleanup = []
    try:
        # Spawn all actors
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(
            world, parked_spots, destination_parking_spot, NUM_RANDOM_CARS
        )
        actors_to_cleanup.extend(parked_cars)
        
        traffic_cones, traffic_cone_bbs = town04_spawn_traffic_cones(
            world, TRAFFIC_CONE_POSITIONS
        )
        actors_to_cleanup.extend(traffic_cones)
        
        walkers, walker_bbs = town04_spawn_walkers(world, WALKER_POSITIONS)
        actors_to_cleanup.extend(walkers)
        
        world.tick()  # Load actors
        
        yield {
            'parked_cars_bbs': parked_cars_bbs,
            'traffic_cone_bbs': traffic_cone_bbs,
            'walker_bbs': walker_bbs,
            'walkers': walkers
        }
    finally:
        for actor in actors_to_cleanup:
            actor.destroy()

def update_perception(car, static_bbs, dynamic_bbs, latency, frame_idx, perception_state):
    """Handle perception updates based on latency."""
    if latency == 0:
        # No latency - immediate perception
        all_bbs = static_bbs + dynamic_bbs
        car.car.obs = union_obstacle_map(
            car.car.obs,
            mask_obstacle_map(
                obstacle_map_from_bbs(all_bbs),
                car.car.cur.x,
                car.car.cur.y
            )
        )
    elif frame_idx % int(latency / 1000 / DELTA_SECONDS) == 0:
        # Latency-based perception
        car.perceive()
        
        if perception_state['response']:
            car.car.obs = perception_state['response']
        
        if perception_state['request']:
            all_bbs = static_bbs + dynamic_bbs
            perception_state['response'] = union_obstacle_map(
                car.car.obs,
                mask_obstacle_map(
                    obstacle_map_from_bbs(all_bbs),
                    perception_state['request'].x,
                    perception_state['request'].y
                )
            )
        
        perception_state['request'] = car.car.cur

def run_scenario(world, destination_parking_spot, parked_spots, latency, ious, recording_file):
    """Run a single parking scenario."""
    car = None
    recording_cam = None
    
    try:
        with spawn_actors(world, destination_parking_spot, parked_spots) as actors:
            # Initialize ego vehicle
            car = town04_spawn_ego_vehicle(world, destination_parking_spot)
            recording_cam = car.init_recording(recording_file)
            
            # Initialize perception with static obstacles
            static_bbs = actors['parked_cars_bbs'] + actors['traffic_cone_bbs']
            car.car.obs = clear_obstacle_map(obstacle_map_from_bbs(static_bbs))
            
            # Save obstacle map visualization
            save_obstacle_map(car.car.obs.obs)
            
            # Run simulation loop
            perception_state = {'request': None, 'response': None}
            frame_idx = 0
            
            while not is_done(car):
                # Update dynamic obstacles
                walker_bbs = update_walkers(actors['walkers'])
                
                # Tick simulation
                world.tick()
                car.localize()
                
                # Update perception
                update_perception(
                    car, static_bbs, walker_bbs, 
                    latency, frame_idx, perception_state
                )
                
                # Replan if needed
                if should_replan(frame_idx, car):
                    car.plan()
                
                # Execute step
                car.run_step()
                car.process_recording_frames(latency=latency)
                frame_idx += 1
            
            # Calculate and record IOU
            iou = car.iou()
            ious.append(iou)
            print(f'IOU: {iou}')
            
    finally:
        if recording_cam:
            recording_cam.destroy()
        if car:
            car.destroy()

def should_replan(frame_idx, car):
    """Determine if replanning is needed."""
    return (frame_idx % REPLAN_INTERVAL == 0 and 
            car.car.cur.distance(car.car.destination) > DISTANCE_THRESHOLD)

def save_obstacle_map(obs_map):
    """Save obstacle map visualization."""
    plt.figure(figsize=(8, 8))
    plt.imshow(obs_map, cmap='gray')
    plt.title('Obstacle Map')
    plt.axis('off')
    plt.savefig('obs_map.png', dpi=100, bbox_inches='tight')
    plt.close()

def plot_iou_results(latency_ious):
    """Create and save IOU scatter plot."""
    plt.figure(figsize=(10, 6))
    
    for latency, ious in latency_ious:
        if len(ious) > 0:
            x_scatter = np.random.normal(loc=latency, scale=0.05, size=len(ious))
            plt.scatter(x_scatter, ious, alpha=0.6, label=f'{latency}ms', s=50)
    
    plt.title('Parking IOU Values vs Perception Latency', fontsize=14)
    plt.xticks(PERCEPTION_LATENCIES, [f'{lat}ms' for lat in PERCEPTION_LATENCIES])
    plt.xlabel('Perception Latency', fontsize=12)
    plt.ylabel('IOU Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('iou_scatter.png', dpi=150)
    plt.close()

def main():
    """Main experiment runner."""
    recording_file = None
    
    try:
        # Initialize simulation
        client = load_client()
        world = town04_load(client)
        town04_spectator_bev(world)
        
        # Setup video recording
        recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
        recording_file.init_video_stream('vp9', fps=VIDEO_FPS)
        
        # Run experiments for each latency configuration
        latency_ious = []
        
        for latency in PERCEPTION_LATENCIES:
            ious = []
            print(f'\n=== Running scenarios for latency: {latency}ms ===')
            
            for destination, parked_spots in SCENARIOS:
                print(f'  Scenario: parking spot {destination}, occupied: {parked_spots}')
                run_scenario(world, destination, parked_spots, latency, ious, recording_file)
            
            latency_ious.append((latency, ious))
            
            # Print summary statistics
            if ious:
                print(f'  Results: mean IOU = {np.mean(ious):.3f}, std = {np.std(ious):.3f}')
        
        # Generate visualization
        plot_iou_results(latency_ious)
        print('\nResults saved to iou_scatter.png')
        
    except KeyboardInterrupt:
        print('\nSimulation interrupted by user')
    except Exception as e:
        print(f'Error during simulation: {e}')
        raise
    finally:
        if recording_file:
            recording_file.close()
        if 'world' in locals():
            world.tick()

if __name__ == '__main__':
    main()