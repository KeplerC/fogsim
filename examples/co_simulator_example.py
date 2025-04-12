import numpy as np
import cv2  # Add OpenCV for video writing
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("cosimulator.log")  # Save logs to file
    ]
)

from cloudsim import GymCoSimulator, CarlaCoSimulator
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator

def run_gym_example():
    """Example of using the Gym co-simulator with NS.py network simulator."""
    try:
        import gym
    except ImportError:
        print("Gym not installed. Please install with: pip install 'cloudsim[gym]'")
        return
    
    # Create network simulator with virtual clock
    network_sim = NSPyNetworkSimulator(
        source_rate=10000.0,  # 10 Kbps
        weights=[1, 2],       # Weight client->server flows lower than server->client
        debug=True
    )
    
    # Create Gym environment with rgb_array render mode
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # Create co-simulator
    co_sim = GymCoSimulator(network_sim, env, timestep=0.01)
    
    # Set up video writer and latency data for plotting
    frames = []
    round_trip_latencies = []
    timesteps = []
    
    # Run simulation
    observation = co_sim.reset()
    print("Initial observation type:", type(observation))
    print("Initial observation:", observation)
    
    # Handle different return types from reset() - could be observation or (observation, info)
    if isinstance(observation, tuple):
        observation = observation[0]  # Extract observation from tuple
    
    done = False
    step_count = 0
    
    while not done and step_count < 200:  # Add back step limit for safety
        # Simple policy: move cart in the direction of pole tilt
        # Pole angle is at index 2 in the observation array
        pole_angle = observation[2]
        action = 1 if pole_angle > 0 else 0
        
        # Record time before step
        start_time = co_sim.get_current_time()
        
        # Step the co-simulator
        step_result = co_sim.step(action)
        
        # Handle different return formats (obs, reward, done, info) or (obs, reward, terminated, truncated, info)
        if len(step_result) == 4:
            observation, reward, done, info = step_result
        else:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        # Record latency information if available in info
        if 'round_trip_latency' in info:
            round_trip_latencies.append(info['round_trip_latency'])
            timesteps.append(start_time)
        
        # Get the rendered frame as RGB array
        frame = co_sim.render()
        if frame is not None:
            frames.append(frame)
            
        step_count += 1
    
    print(f"Collected {len(frames)} frames and {len(round_trip_latencies)} latency measurements")
    print(f"CartPole survived for {step_count} steps")

    co_sim.close()
    
    # Save collected frames as video
    if frames:
        print(f"Saving {len(frames)} frames to simulation_video_gym.mp4")
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('simulation_video_gym.mp4', fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR (OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
        print(f"Video saved successfully with {len(frames)} frames")
    
    # Plot latency data if available
    if timesteps:
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, round_trip_latencies, 'r-', linewidth=2.0, label='Round-trip Latency')
        plt.title("Round-trip Environment-Action Latency")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Latency (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig("round_trip_latency.png")
        plt.show()

if __name__ == "__main__":
    print("Running Gym example with NS.py network simulator...")
    run_gym_example()
