#!/usr/bin/env python3
"""
Simple FogSim Example - Demonstrates the new streamlined API

This example shows how to use the refactored FogSim core with the three modes:
1. Virtual timeline (VIRTUAL) 
2. Real clock + simulated network (SIMULATED_NET)
3. Real clock + real network (REAL_NET)
"""

import numpy as np
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the new FogSim core
from fogsim.core import FogSim, SimulationMode
from fogsim.handlers import GymHandler


def run_episode(fogsim: FogSim, num_steps: int = 50) -> dict:
    """Run a single episode and return metrics."""
    start_time = time.time()
    
    obs, info = fogsim.reset()
    total_reward = 0
    
    for step in range(num_steps):
        # Random policy
        action = fogsim.action_space.sample()
        
        # Step simulation
        obs, reward, success, termination, timeout, info = fogsim.step(action)
        total_reward += reward
        
        if termination or timeout:
            break
    
    wallclock_time = time.time() - start_time
    sim_time = info.get('simulation_time', 0)
    
    return {
        'total_reward': total_reward,
        'steps': step + 1,
        'wallclock_time': wallclock_time,
        'simulation_time': sim_time,
        'fps': (step + 1) / wallclock_time if wallclock_time > 0 else 0,
        'success': success
    }


def main():
    """Demonstrate FogSim three modes."""
    print("="*60)
    print("FogSim Refactored - Three Modes Demo")
    print("="*60)
    
    # Create gym handler
    handler = GymHandler("CartPole-v1")
    
    # Test each mode
    modes = [
        (SimulationMode.VIRTUAL, "Virtual Timeline"),
        (SimulationMode.SIMULATED_NET, "Real Clock + Simulated Network"),  
        (SimulationMode.REAL_NET, "Real Clock + Real Network")
    ]
    
    results = {}
    
    for mode, description in modes:
        print(f"\n--- {description} ---")
        
        try:
            # Create FogSim instance
            fogsim = FogSim(handler, mode=mode, timestep=0.05)
            
            # Run episode
            result = run_episode(fogsim, num_steps=100)
            results[mode] = result
            
            # Print results
            print(f"Reward: {result['total_reward']:.1f}")
            print(f"Steps: {result['steps']}")
            print(f"Wallclock: {result['wallclock_time']:.2f}s")
            print(f"FPS: {result['fps']:.1f}")
            print(f"Success: {result['success']}")
            
            fogsim.close()
            
        except Exception as e:
            print(f"Error in {mode.value} mode: {e}")
            logger.exception(f"Mode {mode.value} failed")
    
    # Comparison
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print("="*60)
        
        baseline_fps = None
        for mode, result in results.items():
            fps = result['fps']
            if baseline_fps is None:
                baseline_fps = fps
                speedup = 1.0
            else:
                speedup = fps / baseline_fps if baseline_fps > 0 else 1.0
            
            print(f"{mode.value:15s}: {fps:6.1f} FPS ({speedup:4.1f}x)")
    
    handler.close()


if __name__ == "__main__":
    main()