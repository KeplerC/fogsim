import numpy as np
import cv2  # Add OpenCV for video writing
import matplotlib.pyplot as plt
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
    co_sim = GymCoSimulator(network_sim, env)
    
    # Set up video writer and latency data for plotting
    frames = []
    action_latencies = []
    observation_latencies = []
    total_latencies = []
    timesteps = []
    
    # Run simulation
    observation = co_sim.reset()
    done = False
    step_count = 0
    
    while not done and step_count < 200:  # Limit to 200 steps for example
        # Random action for example
        action = env.action_space.sample()
        
        # Record time before step
        start_time = co_sim.get_current_time()
        
        # Step the co-simulator
        observation, reward, done, info = co_sim.step(action)
        
        # Record latency information if available in info
        if 'action_latency' in info and 'observation_latency' in info:
            action_latencies.append(info['action_latency'])
            observation_latencies.append(info['observation_latency'])
            total_latencies.append(info['total_latency'])
            timesteps.append(start_time)
        
        # Get the rendered frame as RGB array
        frame = co_sim.render()
        if frame is not None:
            frames.append(frame)
            
        step_count += 1
    
    print(f"Collected {len(frames)} frames and {len(total_latencies)} latency measurements")

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
        plt.plot(timesteps, action_latencies, 'r-', linewidth=2.0, label='Client → Server (Flow 0)')
        plt.plot(timesteps, observation_latencies, 'b-', linewidth=2.0, label='Server → Client (Flow 1)')
        plt.plot(timesteps, total_latencies, 'g--', linewidth=1.5, label='Total Latency')
        plt.title("Network Latency over Time")
        plt.xlabel("Simulation Time (s)")
        plt.ylabel("Latency (s)")
        plt.legend()
        plt.grid(True)
        plt.savefig("network_latency.png")
        plt.show()

if __name__ == "__main__":
    print("Running Gym example with NS.py network simulator...")
    run_gym_example()
