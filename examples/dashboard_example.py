import gym
import logging
import argparse
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("dashboard.log")  # Save logs to file
    ]
)

from cloudsim.environment.gym_co_simulator import GymCoSimulator
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator
from cloudsim.dashboard import start_dashboard, set_simulator

def run_dashboard_example(host='127.0.0.1', port=5000):
    """Example of using the dashboard with the Gym co-simulator and NS.py network simulator."""
    try:
        # Create network simulator with virtual clock
        network_sim = NSPyNetworkSimulator(
            source_rate=1000000.0,  # 10 Mbps
            weights=[1, 2],       # Weight client->server flows lower than server->client
            debug=True
        )
        
        # Create Gym environment with rgb_array render mode
        env = gym.make('CartPole-v1', render_mode="rgb_array")
        
        # Create co-simulator
        co_sim = GymCoSimulator(network_sim, env, timestep=0.01)
        
        # Initialize the simulator
        observation = co_sim.reset()
        print("Initial observation type:", type(observation))
        print("Initial observation:", observation)
        
        # Set up the simulator for the dashboard
        set_simulator(co_sim)
        
        # Start the dashboard
        print(f"Starting dashboard on http://{host}:{port}/")
        print("Use the dashboard to control the simulation")
        start_dashboard(co_sim, host=host, port=port, debug=False)
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        print("Make sure you have installed all required packages:")
        print("  pip install 'cloudsim[gym]'")
        print("  pip install flask")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Cloud Simulation Dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to serve the dashboard (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8000, help='Port to serve the dashboard (default: 8000)')
    
    args = parser.parse_args()
    
    run_dashboard_example(host=args.host, port=args.port) 