#!/usr/bin/env python3
"""
Demonstration of FogSim's Three Operational Modes

This example shows how to use the refactored FogSim in its three different modes:
1. Virtual Timeline (highest performance, reproducible)
2. Real Clock + Simulated Network (wallclock + ns.py network simulation)
3. Real Clock + Real Network (wallclock + local network simulation)
"""

import numpy as np
import time
import argparse
from typing import Dict

from fogsim import FogSim, SimulationMode
from fogsim.handlers import GymHandler


def run_episode(fogsim: FogSim, num_steps: int = 100) -> Dict[str, float]:
    """Run a single episode and collect metrics."""
    obs, info = fogsim.reset()
    
    metrics = {
        'total_reward': 0.0,
        'steps': 0,
        'success': False,
        'network_delays': [],
        'frame_times': []
    }
    
    start_time = time.time()
    last_frame_time = start_time
    
    for step in range(num_steps):
        # Simple policy: random actions
        action = fogsim.action_space.sample()
        
        # Step environment
        obs, reward, success, termination, timeout, info = fogsim.step(action)
        
        # Track frame time
        current_time = time.time()
        frame_time = current_time - last_frame_time
        last_frame_time = current_time
        
        metrics['frame_times'].append(frame_time)
        metrics['total_reward'] += reward
        metrics['steps'] = step + 1
        
        # Extract network delays if available
        if 'network_latencies' in info:
            for latency_info in info['network_latencies']:
                metrics['network_delays'].append(latency_info.get('latency', 0))
        
        if termination or timeout:
            metrics['success'] = success
            break
    
    metrics['episode_time'] = time.time() - start_time
    metrics['avg_frame_time'] = np.mean(metrics['frame_times']) if metrics['frame_times'] else 0
    metrics['avg_network_delay'] = np.mean(metrics['network_delays']) if metrics['network_delays'] else 0.0
    
    return metrics


def compare_modes(num_episodes: int = 5):
    """Compare performance across all three modes."""
    
    print("\nComparing FogSim's three operational modes:")
    print("1. Virtual Timeline (FogSIM) - Highest performance, perfect reproducibility")
    print("2. Real Clock + Simulated Network - Wallclock synced with ns.py")
    print("3. Real Clock + Real Network - Wallclock with local network simulation")
    
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
        
        if mode == SimulationMode.REAL_NET:
            print("Real Network Mode: Using local network simulation (no tc required)")
        
        for episode in range(num_episodes):
            # Create handler and FogSim instance
            handler = GymHandler(env_name="CartPole-v1")
            fogsim = FogSim(
                handler=handler,
                mode=mode,
                timestep=0.1
            )
            
            # Run episode
            print(f"  Episode {episode + 1}/{num_episodes}...", end='', flush=True)
            metrics = run_episode(fogsim, num_steps=200)
            results[mode].append(metrics)
            
            fps = 1.0 / metrics['avg_frame_time'] if metrics['avg_frame_time'] > 0 else 0
            print(f" Done - Reward: {metrics['total_reward']:.2f}, "
                  f"FPS: {fps:.1f}, "
                  f"Delay: {metrics['avg_network_delay']*1000:.1f}ms")
            
            fogsim.close()
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    baseline_fps = None
    
    for mode, mode_results in results.items():
        if not mode_results:
            continue
            
        avg_reward = np.mean([m['total_reward'] for m in mode_results])
        fps_values = [1.0/m['avg_frame_time'] for m in mode_results if m['avg_frame_time'] > 0]
        avg_fps = np.mean(fps_values) if fps_values else 0
        avg_delay = np.mean([m['avg_network_delay'] for m in mode_results]) * 1000
        
        # Calculate speedup relative to first mode
        if baseline_fps is None:
            baseline_fps = avg_fps
            speedup = 1.0
        else:
            speedup = avg_fps / baseline_fps if baseline_fps > 0 else 1.0
        
        print(f"\n{mode.value.upper()}:")
        print(f"  Average Reward: {avg_reward:.2f}")
        print(f"  Average FPS: {avg_fps:.1f} ({speedup:.2f}x speedup)")
        print(f"  Average Network Delay: {avg_delay:.1f}ms")
        
        # Check reproducibility for virtual mode
        if mode == SimulationMode.VIRTUAL:
            rewards = [m['total_reward'] for m in mode_results]
            steps = [m['steps'] for m in mode_results]
            
            # Check if all runs are identical (or very close for floating point)
            reward_variance = np.var(rewards)
            step_variance = np.var(steps)
            
            if reward_variance < 1e-6 and step_variance < 1e-6:
                print(f"  Reproducibility: ✓ PERFECT (variance < 1e-6)")
                print(f"    - All episodes: ~{rewards[0]:.2f} reward, ~{steps[0]} steps")
            else:
                print(f"  Reproducibility: Reward variance = {reward_variance:.4f}")
        
        # Show frame rate advantage for virtual mode
        if mode == SimulationMode.VIRTUAL:
            print(f"  Key Advantage: Decoupled from wallclock time")
            print(f"    - Enables maximum simulation speed")
            print(f"    - Perfect reproducibility for research")


def demonstrate_performance_scaling():
    """Demonstrate performance scaling across modes.""" 
    print(f"\n{'='*60}")
    print("PERFORMANCE SCALING DEMONSTRATION")
    print(f"{'='*60}")
    print("\nTesting same workload across all modes:")
    
    workload_steps = 1000
    
    for mode in SimulationMode:
        print(f"\n--- {mode.value.upper()} MODE ---")
        
        handler = GymHandler(env_name="CartPole-v1")
        fogsim = FogSim(handler, mode=mode, timestep=0.05)  # Faster timestep
        
        start_time = time.time()
        obs, info = fogsim.reset()
        
        for step in range(workload_steps):
            action = fogsim.action_space.sample()
            obs, reward, success, term, timeout, info = fogsim.step(action)
            
            if term or timeout:
                break
        
        elapsed_time = time.time() - start_time
        sim_time = workload_steps * 0.05  # simulation time
        
        print(f"  Wallclock time: {elapsed_time:.2f}s")
        print(f"  Simulation time: {sim_time:.2f}s") 
        print(f"  Speedup factor: {sim_time/elapsed_time:.2f}x")
        print(f"  Steps completed: {step + 1}")
        
        fogsim.close()


def main():
    parser = argparse.ArgumentParser(description="FogSim Three Modes Demo")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes per mode")
    parser.add_argument("--mode", type=str, choices=["all", "virtual", "simulated", "real"],
                        default="all", help="Which mode(s) to test")
    parser.add_argument("--workload", type=int, default=1000,
                        help="Steps for performance test")
    
    args = parser.parse_args()
    
    if args.mode == "all":
        compare_modes(args.episodes)
        demonstrate_performance_scaling()
    else:
        # Run single mode
        mode_map = {
            "virtual": SimulationMode.VIRTUAL,
            "simulated": SimulationMode.SIMULATED_NET,
            "real": SimulationMode.REAL_NET
        }
        mode = mode_map[args.mode]
        
        print(f"Running in {mode.value} mode...")
        
        handler = GymHandler(env_name="CartPole-v1")
        fogsim = FogSim(handler, mode=mode, timestep=0.1)
        
        metrics = run_episode(fogsim, num_steps=500)
        
        print(f"\nResults:")
        print(f"  Total Reward: {metrics['total_reward']:.2f}")
        fps = 1.0/metrics['avg_frame_time'] if metrics['avg_frame_time'] > 0 else 0
        print(f"  Average FPS: {fps:.1f}")
        print(f"  Average Network Delay: {metrics['avg_network_delay']*1000:.1f}ms")
        print(f"  Episode Time: {metrics['episode_time']:.2f}s")
        print(f"  Success: {metrics['success']}")
        
        fogsim.close()


if __name__ == "__main__":
    print("\nFogSim Three Modes Demonstration")
    print("Refactored API - Based on CLAUDE.md evaluation criteria")
    print("-" * 70)
    main()