import argparse
import os
import time
import logging
import statistics
import carla
import numpy as np
import cv2
from configs import EXPERIMENT_CONFIGS
from utils import CollisionDetector
from fogsim.network.nspy_simulator import NSPyNetworkSimulator
from fogsim import CarlaCoSimulator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CarlaSimulator:
    def __init__(self, config):
        self.config = config
        self.client = None
        self.world = None
        self.original_settings = None
        self.ego_vehicle = None
        self.obstacle_vehicle = None
        self.collision_sensor = None
        self.has_collided = False
        self.tick = 0
        self.visualization_enabled = False
        self.top_camera = None
        self.video_writer = None
        self.camera_frames = []
        
    def connect_to_carla(self):
        self.client = carla.Client(self.config['host'], self.config['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world("Town03")
        
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.config['delta_seconds']
        settings.synchronous_mode = self.config['synchronous_mode']
        settings.no_rendering_mode = False
        self.world.apply_settings(settings)
        
    def spawn_actors(self):
        blueprint_library = self.world.get_blueprint_library()
        
        # Get spawn points from the map
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in the map")
            
        # Use first spawn point as base
        base_spawn_point = spawn_points[0]
        
        # Spawn ego vehicle with offset from base spawn point
        ego_bp = blueprint_library.find('vehicle.tesla.model3')
        ego_offset = self.config['ego_spawn_offset']
        ego_transform = carla.Transform(
            carla.Location(
                x=base_spawn_point.location.x + ego_offset['x'],
                y=base_spawn_point.location.y + ego_offset['y'],
                z=base_spawn_point.location.z + ego_offset.get('z', 0.0)
            ),
            carla.Rotation(
                yaw=base_spawn_point.rotation.yaw + ego_offset.get('yaw', 0.0)
            )
        )
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_transform)
        if self.ego_vehicle is None:
            raise RuntimeError(f"Failed to spawn ego vehicle with offset {ego_offset}")
        
        # Spawn obstacle vehicle with offset from base spawn point
        obstacle_bp = blueprint_library.find('vehicle.lincoln.mkz_2020')
        obstacle_offset = self.config['obstacle_spawn_offset']
        obstacle_transform = carla.Transform(
            carla.Location(
                x=base_spawn_point.location.x + obstacle_offset['x'],
                y=base_spawn_point.location.y + obstacle_offset['y'],
                z=base_spawn_point.location.z + obstacle_offset.get('z', 0.0)
            ),
            carla.Rotation(
                yaw=base_spawn_point.rotation.yaw + obstacle_offset.get('yaw', 0.0)
            )
        )
        self.obstacle_vehicle = self.world.try_spawn_actor(obstacle_bp, obstacle_transform)
        if self.obstacle_vehicle is None:
            raise RuntimeError("Failed to spawn obstacle vehicle")
        
        # Add collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=self.ego_vehicle
        )
        self.collision_sensor.listen(lambda event: self._on_collision(event))
        
        # Add top-down camera for visualization if enabled
        if self.visualization_enabled:
            self._spawn_top_camera()
        
    def _on_collision(self, event):
        self.has_collided = True
        
    def reset(self):
        self.cleanup()
        self.has_collided = False
        self.tick = 0
        self.connect_to_carla()
        self.spawn_actors()
        self.world.tick()
        
    def step(self, brake=False):
        if brake:
            control = carla.VehicleControl(throttle=0.0, brake=1.0)
        else:
            control = carla.VehicleControl(throttle=self.config['throttle'], steer=0.0)
        
        self.ego_vehicle.apply_control(control)
        
        # Keep obstacle vehicle stationary
        obstacle_control = carla.VehicleControl(throttle=0.0, brake=1.0)
        self.obstacle_vehicle.apply_control(obstacle_control)
        
        self.world.tick()
        self.tick += 1
        
        observation = self.get_observation()
        if self.visualization_enabled and observation:
            self.update_visualization(observation)
        return observation
    
    def get_observation(self):
        if self.ego_vehicle is None or self.obstacle_vehicle is None:
            return None
            
        ego_transform = self.ego_vehicle.get_transform()
        obstacle_transform = self.obstacle_vehicle.get_transform()
        
        return {
            'ego_pos': [ego_transform.location.x, ego_transform.location.y],
            'obstacle_pos': [obstacle_transform.location.x, obstacle_transform.location.y],
            'tick': self.tick,
            'has_collided': self.has_collided
        }
    
    def enable_visualization(self):
        self.visualization_enabled = True
        self.camera_frames = []
        
        # Create output directory for video
        os.makedirs('video_output', exist_ok=True)
        
        # Initialize video writer (will be set up when first frame is received)
        self.video_writer = None
        
        logger.info("Top-down camera visualization enabled")
        
    def _spawn_top_camera(self):
        """Spawn a top-down camera to record the simulation"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Create top-down camera
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        # Get base spawn point for reference
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available for camera reference")
        base_spawn_point = spawn_points[0]
        
        # Position camera with offset from base spawn point
        camera_offset = self.config.get('camera_offset', {})
        camera_transform = carla.Transform(
            carla.Location(
                x=base_spawn_point.location.x + camera_offset.get('x', 10.0), 
                y=base_spawn_point.location.y + camera_offset.get('y', 0.0), 
                z=base_spawn_point.location.z + camera_offset.get('z', 50.0)
            ),
            carla.Rotation(pitch=-90.0)  # Looking straight down
        )
        
        self.top_camera = self.world.spawn_actor(camera_bp, camera_transform)
        
        # Set up camera callback
        def camera_callback(image):
            self._process_camera_frame(image)
        
        self.top_camera.listen(camera_callback)
        logger.info("Top-down camera spawned and listening")
        
    def _process_camera_frame(self, image):
        """Process incoming camera frames"""
        if not self.visualization_enabled:
            return
            
        # Convert CARLA image to numpy array
        image.convert(carla.ColorConverter.Raw)
        img_array = np.frombuffer(image.raw_data, dtype=np.uint8)
        img_array = img_array.reshape((image.height, image.width, 4))
        frame_bgr = img_array[:, :, :3]  # Remove alpha channel
        frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        
        # Add text overlay with information
        ego_pos = [0, 0]
        obstacle_pos = [0, 0]
        distance = 0
        
        if hasattr(self, 'ego_vehicle') and self.ego_vehicle:
            ego_transform = self.ego_vehicle.get_transform()
            ego_pos = [ego_transform.location.x, ego_transform.location.y]
            
        if hasattr(self, 'obstacle_vehicle') and self.obstacle_vehicle:
            obstacle_transform = self.obstacle_vehicle.get_transform()
            obstacle_pos = [obstacle_transform.location.x, obstacle_transform.location.y]
            
        distance = np.sqrt((ego_pos[0] - obstacle_pos[0])**2 + (ego_pos[1] - obstacle_pos[1])**2)
        
        # Add text overlay
        cv2.putText(frame_bgr, f'Distance: {distance:.2f}m', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f'Tick: {self.tick}', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.has_collided:
            cv2.putText(frame_bgr, 'COLLISION!', (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        
        # Initialize video writer on first frame
        if self.video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                'video_output/simulation.mp4', fourcc, 20.0, 
                (frame_bgr.shape[1], frame_bgr.shape[0])
            )
            logger.info("Video writer initialized")
        
        # Write frame to video
        self.video_writer.write(frame_bgr)
        
        # Store frame for later processing if needed
        self.camera_frames.append(frame_bgr)
    
    def update_visualization(self, observation):
        """Update visualization - camera automatically records frames"""
        if not self.visualization_enabled:
            return
        # Camera callback handles frame recording automatically
        pass
    
    def cleanup(self):
        try:
            if self.visualization_enabled:
                if self.top_camera:
                    self.top_camera.stop()
                    self.top_camera.destroy()
                    self.top_camera = None
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None
                    logger.info("Video saved to video_output/simulation.mp4")
        except Exception as e:
            logger.warning(f"Error cleaning up visualization: {e}")
            
        try:
            if self.collision_sensor:
                self.collision_sensor.stop()
                self.collision_sensor.destroy()
                self.collision_sensor = None
        except Exception as e:
            logger.warning(f"Error cleaning up collision sensor: {e}")
            
        try:
            if self.ego_vehicle:
                self.ego_vehicle.destroy()
                self.ego_vehicle = None
        except Exception as e:
            logger.warning(f"Error cleaning up ego vehicle: {e}")
            
        try:
            if self.obstacle_vehicle:
                self.obstacle_vehicle.destroy()
                self.obstacle_vehicle = None
        except Exception as e:
            logger.warning(f"Error cleaning up obstacle vehicle: {e}")
            
        try:
            if self.world and self.original_settings:
                self.world.apply_settings(self.original_settings)
        except Exception as e:
            logger.warning(f"Error restoring world settings: {e}")

def run_trial(config, bandwidth, mode, enable_vis=False):
    """Run a single trial with given configuration"""
    simulator = CarlaSimulator(config)
    
    # Setup network simulator
    network_sim = NSPyNetworkSimulator(
        source_rate=bandwidth,
        weights=[1, 1]
    )
    
    # Initialize collision detector
    collision_detector = CollisionDetector(threshold=config['collision_threshold'])
    
    # Initialize co-simulator for network simulation
    co_sim = CarlaCoSimulator(network_sim, simulator, timestep=config['delta_seconds'])
    
    # Enable visualization if requested (before reset)
    if enable_vis:
        simulator.enable_visualization()
    
    # Reset simulation
    simulator.reset()
    
    collisions = 0
    total_steps = config['max_steps']
    brake_applied = False
    
    last_observation = None
    steps_without_observation = 0
    
    for step in range(total_steps):
        # Step with network simulation
        observation = co_sim.step(brake_applied)
        brake_applied = False  # Reset brake flag
        
        if observation is not None:
            last_observation = observation
            steps_without_observation = 0
        else:
            # No observation received due to network delay
            steps_without_observation += 1
            if last_observation is None:
                continue
            # Use last known observation but with increased uncertainty
            observation = last_observation
            
        # Check for collision
        if observation['has_collided']:
            collisions = 1
            logger.info(f"Collision detected at step {step}")
            break
            
        # Get positions
        ego_pos = observation['ego_pos']
        obstacle_pos = observation['obstacle_pos']
        distance = np.sqrt((ego_pos[0] - obstacle_pos[0])**2 + (ego_pos[1] - obstacle_pos[1])**2)
        
        # Emergency braking logic - effectiveness depends on observation freshness
        if distance < config['collision_threshold']:
            # With network delay, brake response is delayed
            if steps_without_observation <= 5:  # Recent observation
                logger.info(f"Emergency brake at step {step}, distance: {distance:.2f}m (fresh data)")
                brake_applied = True
            else:
                # Stale observation - brake anyway but may be too late
                logger.info(f"Emergency brake at step {step}, distance: {distance:.2f}m (stale data, delay={steps_without_observation})")
                brake_applied = True
            
        # Show progress
        if step % 100 == 0:
            logger.info(f"Step {step}: Distance = {distance:.2f}m, Brake = {brake_applied}, Delay = {steps_without_observation}")
            
    if enable_vis:
        logger.info("Simulation completed, video saved")
        
    simulator.cleanup()
    return collisions

def run_experiment(config, bandwidth, mode, num_trials, enable_vis=False):
    """Run multiple trials and calculate collision statistics"""
    collision_rates = []
    
    logger.info(f"Running {num_trials} trials for {mode} mode with {bandwidth} Mbps bandwidth")
    
    for trial in range(num_trials):
        logger.info(f"Trial {trial + 1}/{num_trials}")
        # Only enable visualization for the first trial
        vis_enabled = enable_vis and trial == 0
        collisions = run_trial(config, bandwidth, mode, vis_enabled)
        collision_rates.append(collisions)
        
        # Small delay between trials
        time.sleep(1)
    
    collision_rate = sum(collision_rates) / num_trials
    variance = statistics.variance(collision_rates) if len(collision_rates) > 1 else 0
    
    return collision_rate, variance

def main():
    parser = argparse.ArgumentParser(description='CARLA Sync/Async Collision Comparison')
    parser.add_argument('--bandwidth', type=float, default=10.0, 
                       help='Network bandwidth in Mbps')
    parser.add_argument('--trials', type=int, default=10, 
                       help='Number of trials to run')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Output directory for results')
    parser.add_argument('--visualize', action='store_true',
                       help='Enable visualization for the first trial')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiments for both modes
    results = {}
    
    for mode in ['sync', 'async']:
        config = EXPERIMENT_CONFIGS[mode].copy()
        collision_rate, variance = run_experiment(config, args.bandwidth, mode, args.trials, args.visualize)
        
        results[mode] = {
            'collision_rate': collision_rate,
            'variance': variance
        }
        
        logger.info(f"{mode.upper()} Mode - Collision Rate: {collision_rate:.3f}, Variance: {variance:.6f}")
    
    # Save results
    results_file = os.path.join(args.output_dir, f'results_bandwidth_{args.bandwidth}Mbps.txt')
    with open(results_file, 'w') as f:
        f.write(f"Network Bandwidth: {args.bandwidth} Mbps\n")
        f.write(f"Number of Trials: {args.trials}\n\n")
        
        for mode in ['sync', 'async']:
            f.write(f"{mode.upper()} Mode:\n")
            f.write(f"  Collision Rate: {results[mode]['collision_rate']:.3f}\n")
            f.write(f"  Variance: {results[mode]['variance']:.6f}\n\n")
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT RESULTS")
    print("="*50)
    print(f"Bandwidth: {args.bandwidth} Mbps")
    print(f"Trials: {args.trials}")
    print()
    
    for mode in ['sync', 'async']:
        print(f"{mode.upper()} Mode:")
        print(f"  Collision Rate: {results[mode]['collision_rate']:.3f}")
        print(f"  Variance: {results[mode]['variance']:.6f}")
        print()
    
    variance_diff = results['async']['variance'] - results['sync']['variance']
    print(f"Variance Difference (Async - Sync): {variance_diff:.6f}")
    print(f"Results saved to: {results_file}")

if __name__ == '__main__':
    main()