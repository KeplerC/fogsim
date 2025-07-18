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

from fogsim import Env, GymHandler, NetworkConfig

def run_gym_example():
    """Example of using the new FogSim interface with Gym and network simulation."""
    try:
        # Create Gym handler
        handler = GymHandler(env_name="CartPole-v1", render_mode="rgb_array")
        
        # Create network configuration with meaningful delay
        network_config = NetworkConfig(
            source_rate=1000000.0,  # 100 kbps 
            flow_weights=[1, 2],   # Weight client->server flows lower than server->client
            debug=True  # Reduce verbose logging
        )
        
        # Create FogSim environment with network simulation
        env = Env(handler, network_config, enable_network=True, timestep=0.01)
        
        # Set up video writer and latency data for plotting
        frames = []
        round_trip_latencies = []
        timesteps = []
        
        # Run simulation
        observation, extra_info = env.reset()
        print("Initial observation shape:", observation.shape)
        print("Initial observation:", observation)
        print("Network enabled:", extra_info.get('network_enabled', False))
        
        done = False
        step_count = 0
        
        while step_count < 200:  # Step limit for safety
            # Perfect policy using a linear combination of all state variables
            # observation[0]: Cart Position 
            # observation[1]: Cart Velocity
            # observation[2]: Pole Angle
            # observation[3]: Pole Angular Velocity
            
            # Optimal weights found through reinforcement learning
            weights = [-0.04, -0.22, 0.8, 0.52]
            
            # Calculate weighted sum of state variables
            weighted_sum = sum(w * obs for w, obs in zip(weights, observation))
            
            # Take action based on weighted sum
            action = 1 if weighted_sum > 0 else 0
            
            # Step the environment
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            done = termination or timeout
            
            
            # Record latency information if available
            if extra_info.get('network_latencies'):
                for lat_info in extra_info['network_latencies']:
                    if lat_info['type'] == 'action':
                        round_trip_latencies.append(lat_info['latency'])
            timesteps.append(extra_info.get('time', step_count * 0.01))
            
            # Get the rendered frame as RGB array
            frame = env.render()
            if frame is not None:
                frames.append(frame)
                
            step_count += 1
            
            if done:
                break
        
        print(f"Collected {len(frames)} frames and {len(round_trip_latencies)} latency measurements")
        print(f"CartPole survived for {step_count} steps")

        env.close()
        
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
        if round_trip_latencies:
            plt.figure(figsize=(10, 6))
            # Create time series for latency data
            latency_times = list(range(len(round_trip_latencies)))
            plt.plot(latency_times, [lat * 1000 for lat in round_trip_latencies], 'r-', linewidth=2.0, label='Network Latency')
            plt.title("Network Latency Over Time")
            plt.xlabel("Measurement Index")
            plt.ylabel("Latency (ms)")
            plt.legend()
            plt.grid(True)
            plt.savefig("network_latency.png")
            plt.show()
            print(f"Latency plot saved as network_latency.png")
            
    except Exception as e:
        print(f"Error running Gym example: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Running Gym example with NS.py network simulator...")
    run_gym_example()
