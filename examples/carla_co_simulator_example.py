import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import os
import time
import random
import carla

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("carla_cosimulator.log")  # Save logs to file
    ]
)

from cloudsim import CarlaCoSimulator
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator



# Create a simple vehicle controller wrapper for Carla
class SimpleCarlaEnv:
    """A simple wrapper around Carla for this example."""
    def __init__(self, world):
        self.world = world
        self.vehicle = None
        self.camera = None
        self.sensor_data = None
        self.setup_vehicle()
        self.setup_camera()
        
    def setup_vehicle(self):
        """Set up a vehicle for the player."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        
        if not spawn_points:
            print("No spawn points available!")
            return
        
        # Try multiple spawn points in random order until one works
        random.shuffle(spawn_points)
        
        for transform in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, transform)
                print(f"Vehicle {self.vehicle.id} spawned at {transform.location}")
                return  # Successfully spawned
            except RuntimeError as e:
                if "collision" in str(e).lower():
                    print(f"Collision at {transform.location}, trying another spot...")
                    continue
                else:
                    print(f"Error spawning vehicle: {e}")
                    raise
        
        # If we get here, we couldn't spawn a vehicle at any point
        print("ERROR: Could not spawn vehicle at any location due to collisions.")
        
    def setup_camera(self):
        """Set up a camera to get observations."""
        if not self.vehicle:
            print("No vehicle to attach camera to!")
            return
            
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        
        # Set up a queue for sensor data
        self.camera.listen(lambda image: self._process_image(image))
        
    def _process_image(self, image):
        """Process and store camera images."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # Drop alpha channel
        self.sensor_data = array
        
    def step(self, action):
        """Take a step in the environment with the given action."""
        # Action is expected to be [throttle, steer]
        if len(action) >= 2:
            throttle, steer = action[0], action[1]
        else:
            throttle, steer = 0.0, 0.0
            
        # Apply control to the vehicle
        control = carla.VehicleControl(
            throttle=float(throttle), 
            steer=float(steer), 
            brake=0.0, 
            hand_brake=False, 
            reverse=False, 
            manual_gear_shift=False, 
            gear=0
        )
        self.vehicle.apply_control(control)
        
        # Tick the world
        self.world.tick()
        
        # Wait for sensor data
        timeout = 0.5  # seconds
        start_time = time.time()
        while self.sensor_data is None:
            time.sleep(0.01)
            if time.time() - start_time > timeout:
                print("WARNING: Timed out waiting for sensor data")
                break
            
        # Return observation (camera image)
        return self.sensor_data if self.sensor_data is not None else np.zeros((480, 640, 3))
        
    def reset(self):
        """Reset the environment."""
        if self.vehicle:
            spawn_points = self.world.get_map().get_spawn_points()
            if spawn_points:
                # Try multiple spawn points until one works
                random.shuffle(spawn_points)
                for transform in spawn_points:
                    try:
                        self.vehicle.set_transform(transform)
                        # Success - stop trying more points
                        break
                    except Exception as e:
                        print(f"Error resetting vehicle position: {e}")
                        continue
            
            # Reset velocity
            # self.vehicle.set_velocity(carla.Vector3D(0, 0, 0))
            # self.vehicle.set_angular_velocity(carla.Vector3D(0, 0, 0))
            
        # Tick world and wait for first observation
        self.world.tick()
        
        # Wait for sensor data with timeout
        retry_count = 0
        max_retries = 100
        timeout_per_retry = 0.01
        
        while self.sensor_data is None and retry_count < max_retries:
            time.sleep(timeout_per_retry)
            retry_count += 1
            
        if self.sensor_data is None:
            print("WARNING: Could not get sensor data after reset")
            
        return self.sensor_data if self.sensor_data is not None else np.zeros((480, 640, 3))
        
    def render(self):
        """Return the current observation for rendering."""
        return self.sensor_data
        
    def close(self):
        """Clean up resources."""
        if self.camera:
            self.camera.destroy()
        if self.vehicle:
            self.vehicle.destroy()

            
def run_carla_example():
    """Example of using the Carla co-simulator with NS.py network simulator."""
    try:
        # Check if Carla is installed
        import carla
    except ImportError:
        print("Carla Python API not found. Please install it first.")
        print("You can find installation instructions at: https://carla.readthedocs.io/")
        return
    
    # Create network simulator with specified parameters
    network_sim = NSPyNetworkSimulator(
        source_rate=10000000.0,  # 10 Mbps bandwidth
        weights=[1, 2],  # Weight client->server flows lower than server->client
        debug=True
    )
    
    print("Setting up Carla client...")
    # Set up Carla client and environment
    # Note: These settings can be modified to match user's Carla setup
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)  # seconds
    
    # Load a specific map
    world = client.get_world()
    # Alternatively, load a new map: world = client.load_world('Town01')
    
    # Set up synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    # Create the Carla environment
    carla_env = SimpleCarlaEnv(world)
    
    # Check if vehicle was created successfully
    if carla_env.vehicle is None:
        print("Failed to create vehicle. Exiting example.")
        return
    
    # Create co-simulator
    co_sim = CarlaCoSimulator(network_sim, carla_env, timestep=0.05)
    
    # Set up video writer for saving frames
    frames = []
    network_latencies = []
    timesteps = []
    
    # Run simulation
    observation = co_sim.reset()
    
    # Run for 200 steps or until manually stopped
    for step_count in range(200):
        print(f"Step {step_count}")
        
        # Simple policy: drive straight with slight turns
        # In a real application, you would use a more sophisticated policy or user input
        throttle = 0.5  # Constant throttle
        steer = 0.2 * np.sin(step_count / 10.0)  # Sinusoidal steering pattern
        action = np.array([throttle, steer])
        
        # Step the co-simulator
        observation = co_sim.step(action)
        
        # Get the rendered frame
        frame = co_sim.render()
        if frame is not None:
            frames.append(frame)
            
        # Print progress
        if step_count % 10 == 0:
            print(f"Completed {step_count} steps")
    
    # Clean up
    co_sim.close()
    
    # Save collected frames as video
    if frames:
        print(f"Saving {len(frames)} frames to simulation_video_carla.mp4")
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('simulation_video_carla.mp4', fourcc, 20.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR (OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Video saved successfully with {len(frames)} frames")

if __name__ == "__main__":
    print("Running Carla example with NS.py network simulator...")
    run_carla_example() 