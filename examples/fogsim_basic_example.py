"""Basic FogSim example demonstrating the new handler-based interface.

This example shows how to use the new FogSim interface with handlers
for different simulator types and network configuration.
"""

import numpy as np
from fogsim import Env, GymHandler, NetworkConfig, get_low_latency_config


def basic_gym_example():
    """Basic example using Gym handler without network simulation."""
    print("=== Basic Gym Example (No Network) ===")
    
    try:
        # Create a Gym handler
        handler = GymHandler(env_name="CartPole-v1")
        
        # Create FogSim environment without network simulation
        env = Env(handler, enable_network=False)
        
        # Reset environment
        observation, extra_info = env.reset()
        print(f"Initial observation shape: {observation.shape}")
        print(f"Environment info keys: {list(extra_info.keys())}")
        
        # Run a few steps
        total_reward = 0
        for step in range(10):
            # Random action
            action = np.random.choice([0, 1])  # CartPole has discrete actions
            
            # Step environment
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            total_reward += reward
            
            print(f"Step {step + 1}: reward={reward:.3f}, done={termination or timeout}")
            
            if termination or timeout:
                break
        
        print(f"Total reward: {total_reward:.3f}")
        
        # Clean up
        env.close()
        print("Environment closed successfully")
        
    except ImportError as e:
        print(f"Gym not available: {e}")
        print("Install with: pip install 'fogsim[gym]'")
    except Exception as e:
        print(f"Error running example: {e}")
        import traceback
        traceback.print_exc()


def network_simulation_example():
    """Example with network simulation enabled."""
    print("\n=== Network Simulation Example ===")
    
    try:
        # Create a Gym handler
        handler = GymHandler(env_name="CartPole-v1")
        
        # Create network configuration
        network_config = get_low_latency_config()
        print(f"Network config: {network_config.source_rate} bytes/s, "
              f"{network_config.topology.link_delay*1000:.1f}ms delay")
        
        # Create FogSim environment with network simulation
        env = Env(handler, network_config, enable_network=True)
        
        # Reset environment
        observation, extra_info = env.reset()
        print(f"Network enabled: {extra_info['network_enabled']}")
        
        # Run simulation with network effects
        total_reward = 0
        network_latencies = []
        
        for step in range(20):
            # Random action
            action = np.random.choice([0, 1])
            
            # Step environment (with network simulation)
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            total_reward += reward
            
            # Track network latencies
            if extra_info.get('network_latencies'):
                network_latencies.extend(extra_info['network_latencies'])
            
            print(f"Step {step + 1}: reward={reward:.3f}, "
                  f"network_delay={extra_info.get('network_delay', False)}")
            
            if termination or timeout:
                break
        
        print(f"Total reward: {total_reward:.3f}")
        print(f"Network latencies recorded: {len(network_latencies)}")
        
        if network_latencies:
            avg_latency = np.mean([lat['latency'] for lat in network_latencies])
            print(f"Average network latency: {avg_latency*1000:.2f}ms")
        
        # Clean up
        env.close()
        print("Environment closed successfully")
        
    except ImportError as e:
        print(f"Gym not available: {e}")
        print("Install with: pip install 'fogsim[gym]'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def custom_network_config_example():
    """Example with custom network configuration."""
    print("\n=== Custom Network Configuration Example ===")
    
    try:
        # Create a Gym handler
        handler = GymHandler(env_name="CartPole-v1")
        
        # Create custom network configuration
        network_config = NetworkConfig()
        network_config.source_rate = 1e6  # 1 Mbps
        network_config.flow_weights = [2, 1]  # Prioritize first flow
        network_config.packet_loss_rate = 0.01  # 1% packet loss
        
        # Set high-latency topology (satellite-like)
        network_config.topology.link_delay = 0.3  # 300ms
        network_config.topology.link_bandwidth = 10e6  # 10 Mbps
        
        print(f"Custom network: {network_config.topology.link_delay*1000:.0f}ms delay, "
              f"{network_config.packet_loss_rate*100:.1f}% loss")
        
        # Create environment
        env = Env(handler, network_config, enable_network=True)
        
        # Run simulation
        observation, extra_info = env.reset()
        
        for step in range(15):
            action = np.random.choice([0, 1])
            observation, reward, success, termination, timeout, extra_info = env.step(action)
            
            print(f"Step {step + 1}: reward={reward:.3f}")
            
            if termination or timeout:
                break
        
        env.close()
        print("Custom network simulation completed")
        
    except ImportError as e:
        print(f"Gym not available: {e}")
        print("Install with: pip install 'fogsim[gym]'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def compare_with_without_network():
    """Compare performance with and without network simulation."""
    print("\n=== Network Impact Comparison ===")
    
    try:
        import time
        
        # Test without network
        handler1 = GymHandler(env_name="CartPole-v1")
        env1 = Env(handler1, enable_network=False)
        
        start_time = time.time()
        observation, _ = env1.reset()
        for _ in range(50):
            action = np.random.choice([0, 1])
            observation, reward, success, termination, timeout, extra_info = env1.step(action)
            if termination or timeout:
                break
        no_network_time = time.time() - start_time
        env1.close()
        
        # Test with network
        handler2 = GymHandler(env_name="CartPole-v1")
        network_config = NetworkConfig(source_rate=1000.0)
        env2 = Env(handler2, network_config, enable_network=True)
        
        start_time = time.time()
        observation, _ = env2.reset()
        for _ in range(50):
            action = np.random.choice([0, 1])
            observation, reward, success, termination, timeout, extra_info = env2.step(action)
            if termination or timeout:
                break
        with_network_time = time.time() - start_time
        env2.close()
        
        print(f"Without network: {no_network_time:.3f}s")
        print(f"With network: {with_network_time:.3f}s")
        print(f"Overhead: {((with_network_time - no_network_time) / no_network_time * 100):.1f}%")
        
    except ImportError as e:
        print(f"Gym not available: {e}")
        print("Install with: pip install 'fogsim[gym]'")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all examples
    basic_gym_example()
    network_simulation_example()
    custom_network_config_example()
    compare_with_without_network()
    
    print("\n=== All Examples Completed ===")
    print("To run with CARLA or Mujoco, see carla_example.py and mujoco_example.py")