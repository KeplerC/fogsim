import numpy as np
import cv2  # Add OpenCV for video writing
from cloudsim import GymCoSimulator, CarlaCoSimulator, NS3NetworkSimulator

# Example NS3 network simulator wrapper
class NS3NetworkSimulator:
    def __init__(self):
        self.latency = 0.1  # Fixed latency for example
    
    def get_latency(self) -> float:
        return self.latency
    
    def reset(self):
        pass
    
    def close(self):
        pass

def run_gym_example():
    """Example of using the Gym co-simulator."""
    try:
        import gym
    except ImportError:
        print("Gym not installed. Please install with: pip install 'cloudsim[gym]'")
        return
    
    # Create network simulator
    network_sim = NS3NetworkSimulator()
    
    # Create Gym environment with rgb_array render mode
    env = gym.make('CartPole-v1', render_mode="rgb_array")
    
    # Create co-simulator
    co_sim = GymCoSimulator(network_sim, env)
    
    # Set up video writer
    frames = []
    
    # Run simulation
    observation = co_sim.reset()
    done = False
    
    while not done:
        # Random action for example
        action = env.action_space.sample()
        
        # Step the co-simulator
        observation, reward, done, info = co_sim.step(action)
        
        # Get the rendered frame as RGB array
        frame = co_sim.render()
        if frame is not None:
            frames.append(frame)
    
    print(f"Collected {len(frames)} frames")

    co_sim.close()
    
    # Save collected frames as video
    if frames:
        print(f"Saving {len(frames)} frames to simulation_video.mp4")
        height, width, layers = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter('simulation_video_gym.mp4', fourcc, 30.0, (width, height))
        
        for frame in frames:
            # Convert RGB to BGR (OpenCV uses BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video.write(frame_bgr)
        
        video.release()
    print(f"Video saved successfully with {len(frames)} frames")

if __name__ == "__main__":
    print("Running Gym example...")
    run_gym_example()
