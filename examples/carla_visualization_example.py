#!/usr/bin/env python3
"""
Example demonstrating the CloudSim visualization server with a Carla environment.

This example shows how to:
1. Set up and run the CloudSim visualization server
2. Create a Carla co-simulator
3. Wrap it with visualization capabilities
4. Run a simple simulation loop that sends frames to the visualization server

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

# Import a simple network simulator for demonstration
from simple_network_simulator import SimpleNetworkSimulator

# Import carla
try:
    import carla
    logger.info("Carla module imported successfully")
except ImportError:
    logger.error("Failed to import Carla. Is it installed? Exiting.")
    sys.exit(1)

def run_example(args):
    """Run the visualization example with Carla."""
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
    
    # Create a simple Carla environment wrapper
    logger.info("Creating Carla environment")
    carla_env = SimpleCarlaEnv(client, world)
    
    # Create a simple network simulator
    logger.info("Creating network simulator with latency %d ms", args.latency)
    network_sim = SimpleNetworkSimulator(latency=args.latency / 1000.0)  # convert ms to seconds
    
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
    
    # Run the simulation loop
    logger.info("Starting simulation loop")
    try:
        for step in range(args.steps):
            # Create a simple action (throttle, steer, brake)
            action = np.array([0.5, np.sin(step * 0.1) * 0.5, 0.0])  # Simple driving pattern
            
            # Step the environment
            observation = viz_sim.step(action)
            
            # Log progress every 10 steps
            if step % 10 == 0:
                logger.info(f"Step {step}/{args.steps}")
            
            # Slow down the simulation for better visualization
            time.sleep(args.delay)
            
            # Check if we should stop
            if observation is None:
                logger.warning("Received None observation, stopping simulation")
                break
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt detected, stopping simulation")
    
    finally:
        # Close the environment
        logger.info("Closing environment")
        viz_sim.close()
        
        # Note: The server thread is daemonic and will close automatically when the script exits
        logger.info("Example completed")

class SimpleCarlaEnv:
    """
    Simple Carla environment wrapper for demonstration purposes.
    This is not a full-featured environment, just enough to work with the co-simulator.
    """
    
    def __init__(self, client, world):
        """
        Initialize the Carla environment.
        
        Args:
            client: Carla client instance
            world: Carla world instance
        """
        self.client = client
        self.world = world
        self.vehicle = None
        self.camera = None
        self.camera_image = None
        
        # Set synchronous mode
        self.original_settings = None
        self.setup_world()
        
        logger.info("SimpleCarlaEnv initialized")
    
    def setup_world(self):
        """Set up the world with synchronous mode."""
        # Save original settings
        self.original_settings = self.world.get_settings()
        
        # Set synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(settings)
        
        # Set up a camera sensor
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', '800')
        self.camera_bp.set_attribute('image_size_y', '600')
        self.camera_bp.set_attribute('fov', '90')
        
        logger.info("World set up with synchronous mode")
    
    def reset(self):
        """Reset the environment."""
        # Destroy the old vehicle and camera if they exist
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        
        # Spawn a vehicle
        vehicle_bp = self.world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        # Try multiple spawn points until successful
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:
                    break
            except RuntimeError:
                continue
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle at any spawn point")
        
        # Spawn the camera
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(self.camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Set up the callback to capture images
        self.camera.listen(self._process_image)
        
        # Tick the world a few times to stabilize
        for i in range(10):
            self.world.tick()
            time.sleep(0.1)
        
        logger.info("Environment reset, vehicle and camera spawned")
        
        # Return initial observation
        return self._get_observation()
    
    def step(self, action):
        """
        Step the environment.
        
        Args:
            action: np.array with [throttle, steer, brake]
                   throttle and brake should be between 0 and 1
                   steer should be between -1 and 1
        
        Returns:
            observation: Current observation
        """
        # Handle None action by using default values
        if action is None:
            throttle, steer, brake = 0.0, 0.0, 0.0
        else:
            # Unpack the action
            throttle, steer, brake = action
        
        # Apply the control
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )
        self.vehicle.apply_control(control)
        
        # Tick the world once
        self.world.tick()
        
        # Get the observation
        return self._get_observation()
    
    def _process_image(self, image):
        """Process the received camera image."""
        # Convert to RGB array
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Drop the alpha channel
        self.camera_image = array
    
    def _get_observation(self):
        """Get the current observation."""
        # If camera image is not available yet, return a black image
        if self.camera_image is None:
            return np.zeros((600, 800, 3), dtype=np.uint8)
        return self.camera_image
    
    def render(self, mode='rgb_array'):
        """Render the current observation."""
        # Get the current camera image
        frame = self._get_observation()
        
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
            
            return frame
        return None
    
    def close(self):
        """Clean up resources."""
        if self.camera:
            self.camera.destroy()
            self.camera = None
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None
        
        # Restore original settings
        if self.original_settings:
            self.world.apply_settings(self.original_settings)
        
        logger.info("SimpleCarlaEnv closed")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run CloudSim visualization example with Carla')
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
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_example(args) 