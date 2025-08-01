"""
Real Network Mode Demo for FogSim

This example demonstrates flexible real network configuration:
1. Automatic latency measurement from real endpoints
2. Manual tc configuration (no sudo required)
3. Profile-based network conditions
"""

import argparse
import numpy as np
from typing import Dict

from fogsim import (
    Env, GymHandler, SimulationMode,
    NetworkConfig as NetworkControlConfig
)
from fogsim.real_network import (
    RealNetworkManager, LatencyMeasurer, create_real_network_config
)


def test_latency_measurement():
    """Test latency measurement to various endpoints"""
    print("\n" + "="*70)
    print("LATENCY MEASUREMENT TEST")
    print("="*70)
    
    measurer = LatencyMeasurer()
    targets = {
        "Local": "127.0.0.1",
        "Google DNS": "8.8.8.8",
        "Cloudflare": "1.1.1.1"
    }
    
    for name, target in targets.items():
        try:
            print(f"\nMeasuring {name} ({target})...")
            result = measurer.measure_latency(target, count=5)
            print(f"  Average: {result.avg_latency:.1f}ms")
            print(f"  Min/Max: {result.min_latency:.1f}/{result.max_latency:.1f}ms")
            print(f"  Jitter: {result.jitter:.1f}ms")
            print(f"  Loss: {result.packet_loss:.1f}%")
        except Exception as e:
            print(f"  Failed: {e}")


def run_with_real_network(target: str = None, profile: str = None, 
                         mode: str = "manual", steps: int = 200, args=None):
    """Run simulation with real network configuration"""
    
    print("\n" + "="*70)
    print("REAL NETWORK MODE SIMULATION")
    print("="*70)
    
    # Create real network configuration
    if target:
        print(f"\nMeasuring network to {target}...")
    elif profile:
        print(f"\nUsing network profile: {profile}")
    else:
        print("\nUsing default network configuration")
    
    config, manager = create_real_network_config(
        target=target,
        profile=profile,
        mode=mode,
        interface="lo"  # Use loopback for demo
    )
    
    print(f"\nNetwork Configuration:")
    print(f"  Delay: {config.delay:.1f}ms")
    print(f"  Jitter: {config.jitter:.1f}ms")
    print(f"  Bandwidth: {config.bandwidth:.1f}Mbps" if config.bandwidth else "  Bandwidth: Unlimited")
    print(f"  Loss: {config.loss:.1f}%")
    
    if mode == "manual":
        print("\n[!] Manual mode: Please configure tc yourself using the commands above")
        if args and hasattr(args, 'no_wait') and args.no_wait:
            print("    Continuing without waiting (--no-wait specified)...")
        else:
            try:
                print("    Then press Enter to continue...")
                input()
            except EOFError:
                print("    No input available, continuing...")
    
    # Create environment
    print("\nRunning simulation...")
    handler = GymHandler(env_name="CartPole-v1")
    env = Env(
        handler=handler,
        enable_network=True,
        timestep=0.1,
        mode=SimulationMode.REAL_NET
    )
    
    # Configure network (this just stores the config, actual tc is user-managed in manual mode)
    env.configure_network(config)
    
    # Run episode
    obs, info = env.reset()
    total_reward = 0.0
    network_delays = []
    
    for _ in range(steps):
        action = env.handler.env.action_space.sample()
        obs, reward, success, termination, timeout, info = env.step(action)
        total_reward += reward
        
        # Collect network delays
        if 'network_latencies' in info:
            for latency_info in info['network_latencies']:
                network_delays.append(latency_info['latency'] * 1000)  # Convert to ms
        
        if termination or timeout:
            break
    
    env.close()
    
    # Report results
    print(f"\nSimulation Results:")
    print(f"  Total Reward: {total_reward:.1f}")
    print(f"  Episode Length: {info['step']}")
    if network_delays:
        print(f"  Avg Network Delay: {np.mean(network_delays):.1f}ms")
        print(f"  Delay Std Dev: {np.std(network_delays):.1f}ms")
    
    if mode == "manual":
        print("\n[!] Remember to reset tc configuration:")
        print(f"    sudo tc qdisc del dev lo root")


def demonstrate_profiles():
    """Demonstrate different network profiles"""
    print("\n" + "="*70)
    print("NETWORK PROFILES")
    print("="*70)
    
    manager = RealNetworkManager(mode="manual", interface="lo")
    profiles = ["lan", "wifi", "3g", "4g", "satellite", "edge", "cloud"]
    
    print("\nAvailable network profiles:")
    for profile in profiles:
        config = manager.configure_from_profile(profile)
        print(f"\n{profile.upper()}:")
        print(f"  Delay: {config.delay}ms, Jitter: {config.jitter}ms")
        print(f"  Bandwidth: {config.bandwidth}Mbps, Loss: {config.loss}%")


def main():
    parser = argparse.ArgumentParser(
        description="FogSim Real Network Mode Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test latency measurement
  python real_network_demo.py --test-latency
  
  # Use measured latency from Google DNS
  python real_network_demo.py --target 8.8.8.8
  
  # Use predefined profile
  python real_network_demo.py --profile 4g
  
  # Use automatic tc configuration (requires sudo)
  python real_network_demo.py --profile satellite --mode auto
  
  # Show available profiles
  python real_network_demo.py --show-profiles
        """
    )
    
    parser.add_argument("--target", type=str, help="Measure latency from target host")
    parser.add_argument("--profile", type=str, 
                       choices=["lan", "wifi", "wan", "3g", "4g", "satellite", "edge", "cloud"],
                       help="Use predefined network profile")
    parser.add_argument("--mode", type=str, default="manual", choices=["manual", "auto"],
                       help="TC configuration mode (default: manual)")
    parser.add_argument("--test-latency", action="store_true",
                       help="Test latency measurement")
    parser.add_argument("--show-profiles", action="store_true",
                       help="Show available network profiles")
    parser.add_argument("--steps", type=int, default=200,
                       help="Number of simulation steps")
    parser.add_argument("--no-wait", action="store_true",
                       help="Don't wait for user input in manual mode")
    
    args = parser.parse_args()
    
    if args.test_latency:
        test_latency_measurement()
    elif args.show_profiles:
        demonstrate_profiles()
    else:
        # Run simulation with real network
        run_with_real_network(
            target=args.target,
            profile=args.profile,
            mode=args.mode,
            steps=args.steps,
            args=args
        )
    
    print("\n✓ Real network mode demonstration complete!")


if __name__ == "__main__":
    main()