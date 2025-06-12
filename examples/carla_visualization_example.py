#!/usr/bin/env python3
"""
Example demonstrating the CloudSim visualization server with a Carla environment.

This example shows how to:
1. Set up and run the CloudSim visualization server
2. Create a Carla co-simulator
3. Wrap it with visualization capabilities
4. Run a Monte Carlo simulation with collision detection

Requirements:
- carla (tested with carla==0.9.13)
- Python 3.7+ (tested with Python 3.8)

Usage:
    python carla_visualization_example.py
"""

import os
import sys
import time
import logging
import numpy as np
import threading
import argparse
from filterpy.kalman import KalmanFilter
from scipy.stats import norm

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import CloudSim modules
from cloudsim.visualization.visualization_server import VisualizationServer
from cloudsim.visualization.simulator_wrapper import VisualizationCoSimulator
from cloudsim.environment.carla_co_simulator import CarlaCoSimulator
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator

# Import carla
try:
    import carla
    logger.info("Carla module imported successfully")
except ImportError:
    logger.error("Failed to import Carla. Is it installed? Exiting.")
    sys.exit(1)

# Add after imports
unprotected_right_turn_config = {
    'simulation': {
        'host': 'localhost',
        'port': 2000,
        'delta_seconds': 0.01,  # 100fps
        'emergency_brake_threshold': 1.1,
        'prediction_steps': 8000
    },
    'ego_vehicle': {
        'model': 'vehicle.tesla.model3',
        'spawn_offset': {
            'x': 4,
            'y': -90,
            'yaw': 0
        },
        'go_straight_ticks': 500,  # * 10ms = 5s
        'turn_ticks': 250,  # * 10ms = 2.5s
        'after_turn_ticks': 200,  # Add this new parameter for post-turn straight driving
        'throttle': {
            'straight': 0.4,
            'turn': 0.4,
            'after_turn': 0.4  # Add throttle for after turn
        },
        'steer': {
            'turn': 0.3
        }
    },
    'obstacle_vehicle': {
        'model': 'vehicle.lincoln.mkz_2020',
        'spawn_offset': {
            'x': 19,
            'y': 28,
            'yaw': 90
        },
        'go_straight_ticks': 400,  # * 10ms = 4s
        'turn_ticks': 200,  # * 10ms = 2s
        'after_turn_ticks': 350,  # * 10ms = 3.5s
        'throttle': {
            'straight': 0.52,
            'turn': 0.4,
            'after_turn': 0.5
        },
        'steer': {
            'straight': 0.0,
            'turn': 0.0,
            'after_turn': 0.0
        }
    },
    'video': {
        'width': 800,
        'height': 800
    },
    'camera': {
        'fov': '90',
        'height': 50.0,
        'offset': {
            'x': 0.0,
            'y': 0.0
        }
    }
}

class EKFObstacleTracker:
    """Extended Kalman Filter for tracking obstacle position and velocity."""
    def __init__(self, ego_vehicle, obstacle_vehicle, dt=0.1):
        self.dt = dt
        self.kf = KalmanFilter(dim_x=4, dim_z=2)  # State: [x, y, vx, vy], Measurement: [x, y]
        
        # Initialize state transition matrix
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Initialize measurement matrix
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Initialize covariance matrices
        self.kf.R *= 0.1  # Measurement noise
        self.kf.Q *= 0.1  # Process noise
        self.kf.P *= 100  # Initial state uncertainty
        
        # Initialize state
        self.kf.x = np.zeros(4)
        
    def update(self, measurement, timestamp):
        """Update the filter with a new measurement."""
        self.kf.predict()
        self.kf.update(measurement[:2])  # Only use x,y from measurement
        
    def predict_future_position(self, steps):
        """Predict future positions for a given number of steps."""
        predicted_positions = []
        x = self.kf.x.copy()
        
        for _ in range(steps):
            x = self.kf.F @ x
            predicted_positions.append(x[:2])  # Only return x,y positions
            
        return np.array(predicted_positions)

def calculate_collision_probabilities(tracker, predicted_positions, ego_trajectory, current_tick):
    """Calculate collision probabilities for predicted positions."""
    collision_probabilities = []
    max_prob = 0
    collision_time = None
    
    for i, pos in enumerate(predicted_positions):
        # Calculate distance to ego vehicle (using only x,y coordinates)
        ego_pos = ego_trajectory[current_tick + i][:2]  # Take only x,y coordinates
        distance = np.linalg.norm(pos - ego_pos)
        
        # Simple collision probability based on distance
        prob = norm.pdf(distance, loc=0, scale=2.0)  # 2.0 meters standard deviation
        collision_probabilities.append(prob)
        
        if prob > max_prob:
            max_prob = prob
            collision_time = current_tick + i
            
    return max_prob, collision_time, collision_probabilities

class MonteCarloCarlaEnv:
    """Carla environment for Monte Carlo simulation with visualization."""
    
    def __init__(self, client, world):
        """Initialize the environment."""
        self.client = client
        self.world = world
        self.ego_vehicle = None
        self.obstacle_vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.has_collided = False
        self.frame_queue = []
        self.tick = 0
        self.camera_image = None
        self.config = unprotected_right_turn_config  # Use the same config as main.py
        
        # Set up synchronous mode
        self.original_settings = self.world.get_settings()
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.config['simulation']['delta_seconds']
        self.world.apply_settings(settings)
        
        # Set up camera blueprint
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', str(self.config['video']['width']))
        self.camera_bp.set_attribute('image_size_y', str(self.config['video']['height']))
        self.camera_bp.set_attribute('fov', self.config['camera']['fov'])
        
        logger.info("MonteCarloCarlaEnv initialized")
    
    def reset(self):
        """Reset the environment."""
        self._clean_up_actors()
        
        # Reset simulation state
        self.has_collided = False
        self.frame_queue = []
        self.camera_image = None
        self.tick = 0
        
        # Spawn vehicles
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Setup ego vehicle spawn point
        ego_spawn_point = spawn_points[0]
        ego_spawn_point.location.x += self.config['ego_vehicle']['spawn_offset']['x']
        ego_spawn_point.location.y += self.config['ego_vehicle']['spawn_offset']['y']
        ego_spawn_point.rotation.yaw += self.config['ego_vehicle']['spawn_offset']['yaw']
        
        # Spawn ego vehicle
        ego_bp = blueprint_library.find(self.config['ego_vehicle']['model'])
        ego_bp.set_attribute('role_name', 'ego')
        self.ego_vehicle = self.world.try_spawn_actor(ego_bp, ego_spawn_point)
        logger.info(f"Ego vehicle spawned at {ego_spawn_point.location}")
        
        # Setup obstacle vehicle spawn point relative to ego
        obstacle_spawn_point = spawn_points[1]
        obstacle_spawn_point.location.x = ego_spawn_point.location.x + self.config['obstacle_vehicle']['spawn_offset']['x']
        obstacle_spawn_point.location.y = ego_spawn_point.location.y + self.config['obstacle_vehicle']['spawn_offset']['y']
        obstacle_spawn_point.rotation.yaw = ego_spawn_point.rotation.yaw + self.config['obstacle_vehicle']['spawn_offset']['yaw']
        
        # Spawn obstacle vehicle
        obstacle_bp = blueprint_library.find(self.config['obstacle_vehicle']['model'])
        obstacle_bp.set_attribute('role_name', 'obstacle')
        self.obstacle_vehicle = self.world.try_spawn_actor(obstacle_bp, obstacle_spawn_point)
        logger.info(f"Obstacle vehicle spawned at {obstacle_spawn_point.location}")
        
        # Set up camera for bird's eye view
        camera_transform = carla.Transform(
            carla.Location(
                x=self.config['camera']['offset']['x'],
                y=self.config['camera']['offset']['y'],
                z=self.config['camera']['height']
            ),
            carla.Rotation(
                pitch=-90.0,  # Look straight down
                yaw=0.0,      # No rotation
                roll=0.0      # No roll
            )
        )
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.ego_vehicle)
        logger.info(f"Camera spawned and attached to ego vehicle at {camera_transform.location} with pitch={camera_transform.rotation.pitch}")
        
        # Set up collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego_vehicle)
        logger.info("Collision sensor spawned and attached to ego vehicle")
        
        # Set up callbacks
        self.camera.listen(self._process_image)
        self.collision_sensor.listen(self._process_collision)
        
        # Tick the world a few times to stabilize
        for i in range(10):
            self.world.tick()
            time.sleep(0.1)
            logger.info(f"World tick {i+1}/10")
        
        logger.info("Environment reset, vehicles and sensors spawned")
        return self._get_observation()
    
    def _process_image(self, image):
        """Process camera image."""
        try:
            # Convert to RGB array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            # Convert from BGRA to RGB
            array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
            self.camera_image = array  # Store the latest image
            logger.debug(f"Processed image: shape={array.shape}, dtype={array.dtype}, min={array.min()}, max={array.max()}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
    
    def _process_collision(self, event):
        """Process collision event."""
        self.has_collided = True
    
    def render(self, mode='rgb_array'):
        """Render the current state."""
        try:
            if self.camera_image is not None:
                frame = self.camera_image.copy()  # Use the latest camera image
                # Ensure the frame is in the correct format (uint8 RGB)
                if frame is not None:
                    # Make sure the frame is uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    # Ensure the frame is RGB (3 channels)
                    if len(frame.shape) == 2:  # If grayscale
                        frame = np.stack([frame] * 3, axis=-1)
                    elif frame.shape[2] == 4:  # If RGBA
                        frame = frame[:, :, :3]  # Drop alpha channel
                    
                    logger.debug(f"Rendering frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")
                    return frame
            logger.warning("No frame available, returning black frame")
            return np.zeros((600, 800, 3), dtype=np.uint8)  # Return black frame if no image available
        except Exception as e:
            logger.error(f"Error in render: {str(e)}")
            return np.zeros((600, 800, 3), dtype=np.uint8)
    
    def step(self, action):
        """Step the environment."""
        if action is None:
            brake = False
        else:
            brake = action
            
        # Apply controls based on current tick and brake flag
        if brake:
            # Emergency brake
            ego_control = carla.VehicleControl(throttle=0.0, brake=1.0)
            logger.info("Emergency brake applied")
        else:
            # Normal driving based on config
            ego_control = carla.VehicleControl()
            
            if self.tick < self.config['ego_vehicle']['go_straight_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['straight']
                ego_control.steer = self.config['ego_vehicle']['steer'].get('straight', 0.0)
            elif self.tick < self.config['ego_vehicle']['go_straight_ticks'] + self.config['ego_vehicle']['turn_ticks']:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['turn']
                ego_control.steer = self.config['ego_vehicle']['steer']['turn']
            else:
                ego_control.throttle = self.config['ego_vehicle']['throttle']['after_turn']
                ego_control.steer = self.config['ego_vehicle']['steer'].get('after_turn', 0.0)
        
        # Apply the calculated control
        self.ego_vehicle.apply_control(ego_control)
        
        # Apply obstacle controls
        obstacle_control = carla.VehicleControl()
        if self.tick < self.config['obstacle_vehicle']['go_straight_ticks']:
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['straight']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['straight']
        elif self.tick < self.config['obstacle_vehicle']['go_straight_ticks'] + self.config['obstacle_vehicle']['turn_ticks']:
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['turn']
        else:
            obstacle_control.throttle = self.config['obstacle_vehicle']['throttle']['after_turn']
            obstacle_control.steer = self.config['obstacle_vehicle']['steer']['after_turn']
        
        self.obstacle_vehicle.apply_control(obstacle_control)
        
        # Tick the world
        self.world.tick()
        self.tick += 1
        
        # Log frame status
        logger.debug(f"Camera image available: {self.camera_image is not None}")
        
        return self._get_observation()
    
    def _get_observation(self):
        """Get current observation."""
        if self.ego_vehicle is None or self.obstacle_vehicle is None:
            return None
            
        ego_transform = self.ego_vehicle.get_transform()
        obstacle_transform = self.obstacle_vehicle.get_transform()
        
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
        
        return (ego_pos, obstacle_pos, ego_rot, obstacle_rot, self.tick)
    
    def _clean_up_actors(self):
        """Clean up all actors."""
        if self.collision_sensor:
            self.collision_sensor.destroy()
            self.collision_sensor = None
            
        if self.camera:
            self.camera.destroy()
            self.camera = None
            
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
            self.ego_vehicle = None
            
        if self.obstacle_vehicle:
            self.obstacle_vehicle.destroy()
            self.obstacle_vehicle = None
    
    def close(self):
        """Clean up resources."""
        self._clean_up_actors()
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
        logger.info("MonteCarloCarlaEnv closed")

def run_example(args):
    """Run the Monte Carlo simulation example with visualization."""
    # Set up more detailed logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and start the visualization server
    logger.info("Starting visualization server on port %d", args.port)
    server = VisualizationServer(host=args.host, port=args.port)
    server_thread = server.run_in_thread()
    
    # Wait for server to start
    time.sleep(1)
    
    # Connect to the Carla server
    logger.info(f"Connecting to Carla server at {args.carla_host}:{args.carla_port}")
    try:
        client = carla.Client(args.carla_host, args.carla_port)
        client.set_timeout(args.carla_timeout)
        world = client.get_world()
        logger.info(f"Connected to Carla server, got world {world.get_map().name}")
    except Exception as e:
        logger.error(f"Failed to connect to Carla server: {str(e)}")
        return
    
    # Create the Monte Carlo environment
    logger.info("Creating Monte Carlo environment")
    carla_env = MonteCarloCarlaEnv(client, world)
    
    # Create network simulator
    logger.info("Creating network simulator with latency %d ms", args.latency)
    network_sim = NSPyNetworkSimulator(
        source_rate=100000000.0,  # 100 Mbps
        weights=[1, 2]  # Weight client->server flows lower than server->client
    )
    
    # Create the CarlaCoSimulator
    logger.info("Creating CarlaCoSimulator with timestep %f", args.timestep)
    co_sim = CarlaCoSimulator(
        network_simulator=network_sim,
        carla_env=carla_env,
        timestep=args.timestep
    )
    
    # Wrap with visualization capabilities
    logger.info("Creating visualization wrapper")
    viz_sim = VisualizationCoSimulator(
        co_simulator=co_sim,
        server_url=f"http://{args.host}:{args.port}",
        simulation_id=f"carla_{world.get_map().name}",
        auto_connect=True
    )
    
    # Reset the environment
    logger.info("Resetting environment")
    observation = viz_sim.reset()
    
    # Create the obstacle tracker
    rel_tracker = EKFObstacleTracker(
        carla_env.ego_vehicle,
        carla_env.obstacle_vehicle,
        dt=args.timestep
    )
    
    # Run the simulation loop
    logger.info("Starting simulation loop")
    try:
        for step in range(args.steps):
            # Step the environment
            observation = viz_sim.step(None)  # No brake initially
            
            if observation is not None:
                ego_pos, obstacle_pos, ego_rot, obstacle_rot, tick = observation
                
                # Calculate relative position
                rel_x = obstacle_pos[0] - ego_pos[0]
                rel_y = obstacle_pos[1] - ego_pos[1]
                rel_yaw_deg = obstacle_rot[0] - ego_rot[0]
                rel_yaw_rad = rel_yaw_deg * np.pi / 180.0
                
                # Update tracker
                rel_tracker.update((rel_x, rel_y, rel_yaw_rad), tick)
                
                # Predict future positions
                predicted_positions = rel_tracker.predict_future_position(100)
                
                # Calculate collision probabilities
                ego_trajectory = np.zeros((args.steps, 3))  # Initialize with zeros
                # Update ego trajectory with current position
                ego_trajectory[tick] = ego_pos
                
                max_collision_prob, collision_time, collision_probabilities = calculate_collision_probabilities(
                    rel_tracker, predicted_positions, ego_trajectory, tick)
                
                logger.info(f"Step {step}: Collision probability: {max_collision_prob:.4f}")
                
                # Update metrics in visualization
                metrics = {
                    'collision_probability': f"{max_collision_prob:.4f}",
                    'step': str(step),
                    'ego_position': f"({ego_pos[0]:.2f}, {ego_pos[1]:.2f})",
                    'obstacle_position': f"({obstacle_pos[0]:.2f}, {obstacle_pos[1]:.2f})"
                }
                viz_sim.viz_client.send_metrics(metrics)
                
                # Apply emergency brake if collision probability exceeds threshold
                if max_collision_prob > args.emergency_brake_threshold:
                    logger.info(f"Step {step}: EMERGENCY BRAKE ACTIVATED")
                    observation = viz_sim.step(True)  # Apply brake
                
                if carla_env.has_collided:
                    logger.info(f"Step {step}: Collision detected")
                    break
            
            # Log progress every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}/{args.steps}")
                # Log frame queue status
                logger.info(f"Frame queue size: {len(carla_env.frame_queue)}")
            
            # Slow down the simulation for better visualization
            time.sleep(args.delay)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping simulation")
    
    finally:
        # Close the environment
        logger.info("Closing environment")
        viz_sim.close()
        logger.info("Example completed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Monte Carlo simulation with visualization')
    parser.add_argument('--host', type=str, default='localhost',
                        help='Host address to bind the server to')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to listen on')
    parser.add_argument('--carla-host', type=str, default='localhost',
                        help='Carla server host')
    parser.add_argument('--carla-port', type=int, default=2000,
                        help='Carla server port')
    parser.add_argument('--carla-timeout', type=float, default=10.0,
                        help='Carla client timeout')
    parser.add_argument('--timestep', type=float, default=0.1,
                        help='Simulation timestep in seconds')
    parser.add_argument('--latency', type=int, default=50,
                        help='Network latency in milliseconds')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Number of simulation steps to run')
    parser.add_argument('--delay', type=float, default=0.1,
                        help='Delay between steps for visualization')
    parser.add_argument('--emergency-brake-threshold', type=float, default=0.8,
                        help='Threshold for emergency braking')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_example(args) 