"""
Demonstration of FogSim's Three Operational Modes

This example shows how to use FogSim in its three different modes:
1. Virtual Time (highest performance, reproducible)
2. Simulated Network (wallclock + simulated delays)
3. Real Network (wallclock + real network with Linux tc)
"""

import numpy as np
import time
import argparse
from typing import Dict

from fogsim import (
    Env, GymHandler, SimulationMode,
    NetworkControlConfig, get_low_latency_config
)
from fogsim.real_network import create_real_network_config


def run_episode(env: Env, num_steps: int = 100) -> Dict[str, float]:
    """Run a single episode and collect metrics."""
    _, info = env.reset()
    
    metrics = {
        'total_reward': 0.0,
        'steps': 0,
        'success': False,
        'network_delays': [],
        'frame_times': []
    }
    
    start_time = time.time()
    last_frame_time = start_time
    
    for _ in range(num_steps):
        # Simple policy: random actions
        action = env.handler.env.action_space.sample()
        
        # Step environment
        _, reward, success, termination, timeout, info = env.step(action)
        
        # Track frame time
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time
        
        metrics['frame_times'].append(frame_time)
        metrics['total_reward'] += reward
        metrics['steps'] += 1
        
        # Extract network delays if available
        if 'network_latencies' in info:
            for latency_info in info['network_latencies']:
                metrics['network_delays'].append(latency_info['latency'])
        
        if termination or timeout:
            metrics['success'] = success
            break
    
    metrics['episode_time'] = time.time() - start_time
    metrics['avg_frame_time'] = np.mean(metrics['frame_times'])
    metrics['avg_network_delay'] = np.mean(metrics['network_delays']) if metrics['network_delays'] else 0.0
    
    return metrics


def compare_modes(num_episodes: int = 5):
    """Compare performance across all three modes."""
    
    print("\nComparing FogSim's three operational modes:")
    print("1. Virtual Timeline (FogSIM) - Highest performance, perfect reproducibility")
    print("2. Real Clock + Simulated Network - Wallclock synced with ns.py")
    print("3. Real Clock + Real Network - Wallclock with Linux tc control")
    
    # Network configuration
    network_config = get_low_latency_config()
    
    # Results storage
    results = {
        SimulationMode.VIRTUAL: [],
        SimulationMode.SIMULATED_NET: [],
        SimulationMode.REAL_NET: []
    }
    
    for mode in SimulationMode:
        print(f"\n{'='*60}")
        print(f"Testing Mode: {mode.value.upper()}")
        print(f"{'='*60}")
        
        # Real network mode setup
        if mode == SimulationMode.REAL_NET:
            print("\nReal Network Mode Options:")
            print("1. Manual tc configuration (no sudo required)")
            print("2. Automatic tc configuration (requires sudo)")
            print("\nUsing manual mode for this demo...")
        
        for episode in range(num_episodes):
            # Create environment with specific mode
            handler = GymHandler(env_name="CartPole-v1")
            env = Env(
                handler=handler,
                network_config=network_config,
                enable_network=True,
                timestep=0.1,
                mode=mode
            )
            
            # Configure network for real network mode
            if mode == SimulationMode.REAL_NET:
                # Use manual mode - just show config, user manages tc
                config, manager = create_real_network_config(
                    profile="edge",  # 10ms delay, 100Mbps
                    mode="manual",
                    interface="lo"
                )
                env.configure_network(config)
                print(f"\n  [!] Please configure tc manually or continue without it")
            
            # Run episode
            print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
            metrics = run_episode(env, num_steps=200)
            results[mode].append(metrics)
            
            print(f" Done - Reward: {metrics['total_reward']:.2f}, "
                  f"FPS: {1.0/metrics['avg_frame_time']:.1f}, "
                  f"Delay: {metrics['avg_network_delay']*1000:.1f}ms")
            
            env.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    for mode, mode_results in results.items():
        if not mode_results:
            continue
            
        avg_reward = np.mean([m['total_reward'] for m in mode_results])
        avg_fps = np.mean([1.0/m['avg_frame_time'] for m in mode_results])
        avg_delay = np.mean([m['avg_network_delay'] for m in mode_results]) * 1000
        
        print(f"\n{mode.value.upper()}:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average FPS: {avg_fps:.1f}")
        print(f"  Average Network Delay: {avg_delay:.1f}ms")
        
        # Check reproducibility for virtual mode
        if mode == SimulationMode.VIRTUAL:
            rewards = [m['total_reward'] for m in mode_results]
            steps = [m['steps'] for m in mode_results]
            
            # Check if all runs are identical
            if len(set(rewards)) == 1 and len(set(steps)) == 1:
                print(f"  Reproducibility: ✓ PERFECT (all runs identical)")
                print(f"    - All episodes: {rewards[0]:.0f} reward, {steps[0]} steps")
            else:
                variance = np.var(rewards)
                print(f"  Reproducibility: Variance = {variance:.4f}")
        
        # Show frame rate advantage
        if mode == SimulationMode.VIRTUAL:
            print(f"  Frame Rate Advantage: Can run at maximum speed")
            print(f"    - Not limited by wallclock synchronization")
            print(f"    - Enables faster training and evaluation")


def demonstrate_time_scaling():
    """Demonstrate time scaling capabilities in simulated network mode."""
    print(f"\n{'='*60}")
    print("TIME SCALING DEMONSTRATION")
    print(f"{'='*60}")
    print("\nShowing how simulation speed varies with time scaling:")
    
    handler = GymHandler(env_name="CartPole-v1")
    env = Env(
        handler=handler,
        enable_network=True,
        mode=SimulationMode.SIMULATED_NET
    )
    
    # Test different time scales
    time_scales = [0.5, 1.0, 2.0, 5.0]
    
    for scale in time_scales:
        if hasattr(env.time_manager.backend, 'set_time_scale'):
            env.time_manager.backend.set_time_scale(scale)
            
            env.reset()
            start_time = time.time()
            
            # Run 50 steps
            for _ in range(50):
                action = env.handler.env.action_space.sample()
                env.step(action)
            
            elapsed = time.time() - start_time
            expected = 50 * 0.1 / scale  # 50 steps * 0.1s timestep / scale
            
            print(f"  Scale {scale}x: {elapsed:.2f}s elapsed (expected ~{expected:.2f}s)")
            
            # Show simulation time vs wallclock time
            sim_time_elapsed = 50 * 0.1  # 50 steps * 0.1s timestep
            print(f"    - Simulated {sim_time_elapsed:.1f}s in {elapsed:.2f}s wallclock")
            print(f"    - Effective speedup: {sim_time_elapsed/elapsed:.2f}x")
    
    env.close()


def main():
    parser = argparse.ArgumentParser(description="FogSim Three Modes Demo")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per mode")
    parser.add_argument("--mode", type=str, choices=["all", "virtual", "simulated", "real"],
                        default="all", help="Which mode(s) to test")
    parser.add_argument("--real-target", type=str, default=None,
                        help="Target host for real network measurement")
    parser.add_argument("--real-profile", type=str, default="edge",
                        help="Network profile for real mode")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        compare_modes(args.episodes)
        demonstrate_time_scaling()
    else:
        # Run single mode
        mode_map = {
            "virtual": SimulationMode.VIRTUAL,
            "simulated": SimulationMode.SIMULATED_NET,
            "real": SimulationMode.REAL_NET
        }
        mode = mode_map[args.mode]
        
        handler = GymHandler(env_name="CartPole-v1")
        env = Env(
            handler=handler,
            enable_network=True,
            mode=mode
        )
        
        # Configure real network if in real mode
        if mode == SimulationMode.REAL_NET:
            config, _ = create_real_network_config(
                target=args.real_target,
                profile=args.real_profile,
                mode="manual"
            )
            env.configure_network(config)
        
        print(f"Running in {mode.value} mode...")
        metrics = run_episode(env, num_steps=500)
        
        print(f"\nResults:")
        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        print(f"  Average FPS: {1.0/metrics['avg_frame_time']:.1f}")
        print(f"  Average Network Delay: {metrics['avg_network_delay']*1000:.1f}ms")
        print(f"  Episode Time: {metrics['episode_time']:.2f}s")
        
        env.close()


if __name__ == "__main__":
    print("\nFogSim Three Modes Demonstration")
    print("Based on CLAUDE.md evaluation criteria")
    print("-" * 70)
    main()