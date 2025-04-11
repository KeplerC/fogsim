import numpy as np
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
    
    # Create Gym environment
    env = gym.make('CartPole-v1', render_mode="human")
    
    # Create co-simulator
    co_sim = GymCoSimulator(network_sim, env)
    
    # Run simulation
    observation = co_sim.reset()
    done = False
    
    while not done:
        # Random action for example
        action = env.action_space.sample()
        
        # Step the co-simulator
        observation, reward, done, info = co_sim.step(action)
        
        # Render the environment
        co_sim.render()
    
    co_sim.close()

if __name__ == "__main__":
    print("Running Gym example...")
    run_gym_example()
