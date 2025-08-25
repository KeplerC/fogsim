#!/usr/bin/env python3
"""
Clean script to compare the variance in collision tick numbers between 
synchronous and asynchronous CARLA modes using the configurable sync_mode parameter.
"""

import os
import subprocess
import statistics
import json
import time
import csv
from pathlib import Path

def run_simulation(sync_mode=False, config_type='merge', run_id=0):
    """
    Run a single simulation with specified parameters.
    
    Args:
        sync_mode (bool): Whether to use synchronous mode
        config_type (str): Configuration type ('right_turn', 'left_turn', 'merge')
        run_id (int): Run identifier for output directory
        
    Returns:
        dict: Results including collision tick if collision occurred
    """
    output_dir = f'./sync_comparison/{"sync" if sync_mode else "async"}/run_{run_id}'
    
    # Prepare command
    # Use FogSim to ensure we're using CollisionHandler which has the sync_mode setting
    cmd = [
        'python', 'main.py',
        '--config_type', config_type,
        '--output_dir', output_dir,
        '--no_risk_eval',  # Disable risk evaluation for cleaner comparison
        '--use_fogsim',    # This ensures CollisionHandler is used
        '--cautious_delta_k', '40',
    ]
    
    if sync_mode:
        cmd.append('--sync_mode')
    
    try:
        # Run simulation
        print(f"  Running {'synchronous' if sync_mode else 'asynchronous'} mode, run {run_id}...")
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # Reduced timeout
        
        if result.returncode != 0:
            print(f"    Error: {result.stderr[:100]}")
            return {'success': False, 'error': result.stderr[:100]}
        
        # Check for collision and extract tick
        collision_tick = extract_collision_tick(output_dir)
        
        return {
            'success': True,
            'collision_occurred': collision_tick is not None,
            'collision_tick': collision_tick,
            'output_dir': output_dir
        }
        
    except subprocess.TimeoutExpired as e:
        print(f"    Timeout after 60 seconds")
        print(f"    Partial stdout: {e.stdout[:200] if e.stdout else 'None'}")
        print(f"    Partial stderr: {e.stderr[:200] if e.stderr else 'None'}")
        return {'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"    Exception: {e}")
        return {'success': False, 'error': str(e)}

def extract_collision_tick(output_dir):
    """
    Extract collision tick from simulation output files.
    
    Args:
        output_dir (str): Directory containing simulation output
        
    Returns:
        int or None: Collision tick if found, None otherwise
    """
    # Check statistics file first
    stats_file = Path(output_dir) / 'monte_carlo_results' / 'statistics.csv'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'collision' in row and row['collision'].lower() == 'true':
                        # Found collision, but need to get tick from elsewhere
                        pass
        except Exception:
            pass
    
    # Check collision probabilities file
    for file in Path(output_dir).glob('collision_probabilities*.csv'):
        try:
            with open(file, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:  # Has data beyond header
                    # Get last line before potential collision
                    last_line = lines[-1].strip()
                    if last_line:
                        parts = last_line.split(',')
                        if len(parts) >= 2:
                            tick = int(float(parts[1]))
                            # Heuristic: if simulation stopped early, collision likely occurred
                            # You may need to adjust this based on your simulation length
                            if tick < 900:  # Assuming normal simulation is ~1000 ticks
                                return tick + 1  # Collision likely at next tick
        except Exception:
            pass
    
    # Check simulation output text file if exists
    stats_txt = Path(output_dir) / 'monte_carlo_results' / 'statistics.txt'
    if stats_txt.exists():
        try:
            with open(stats_txt, 'r') as f:
                content = f.read()
                if 'Number of collisions: 1' in content:
                    # Collision occurred but couldn't extract exact tick
                    return -1  # Sentinel value for "collision occurred but tick unknown"
        except Exception:
            pass
    
    return None

def run_comparison(num_runs=10, config_type='merge'):
    """
    Run comparison between synchronous and asynchronous modes.
    
    Args:
        num_runs (int): Number of runs for each mode
        config_type (str): Configuration type to test
        
    Returns:
        dict: Comparison results and statistics
    """
    print(f"Starting Synchronous vs Asynchronous Mode Comparison")
    print(f"Configuration: {config_type}")
    print(f"Number of runs per mode: {num_runs}")
    print("=" * 60)
    
    # Clean up previous results
    os.system('rm -rf ./sync_comparison')
    os.makedirs('./sync_comparison/sync', exist_ok=True)
    os.makedirs('./sync_comparison/async', exist_ok=True)
    
    async_results = []
    sync_results = []
    
    # Run asynchronous simulations
    print("\nRunning ASYNCHRONOUS mode simulations...")
    for i in range(num_runs):
        result = run_simulation(sync_mode=False, config_type=config_type, run_id=i)
        async_results.append(result)
        if result['success'] and result['collision_occurred']:
            print(f"    Collision at tick: {result['collision_tick']}")
        elif result['success']:
            print(f"    No collision")
        time.sleep(2)  # Small delay between runs
    
    # Run synchronous simulations  
    print("\nRunning SYNCHRONOUS mode simulations...")
    for i in range(num_runs):
        result = run_simulation(sync_mode=True, config_type=config_type, run_id=i)
        sync_results.append(result)
        if result['success'] and result['collision_occurred']:
            print(f"    Collision at tick: {result['collision_tick']}")
        elif result['success']:
            print(f"    No collision")
        time.sleep(2)  # Small delay between runs
    
    # Extract collision ticks
    async_ticks = [r['collision_tick'] for r in async_results 
                   if r['success'] and r['collision_occurred'] and r['collision_tick'] > 0]
    sync_ticks = [r['collision_tick'] for r in sync_results 
                  if r['success'] and r['collision_occurred'] and r['collision_tick'] > 0]
    
    # Calculate statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    results = {
        'config_type': config_type,
        'num_runs': num_runs,
        'async': {
            'collision_ticks': async_ticks,
            'num_collisions': len(async_ticks),
            'success_runs': sum(1 for r in async_results if r['success'])
        },
        'sync': {
            'collision_ticks': sync_ticks,
            'num_collisions': len(sync_ticks),
            'success_runs': sum(1 for r in sync_results if r['success'])
        }
    }
    
    print(f"\nAsynchronous Mode:")
    print(f"  Successful runs: {results['async']['success_runs']}/{num_runs}")
    print(f"  Collisions: {results['async']['num_collisions']}")
    if len(async_ticks) > 0:
        print(f"  Collision ticks: {async_ticks}")
    if len(async_ticks) > 1:
        results['async']['mean'] = statistics.mean(async_ticks)
        results['async']['stdev'] = statistics.stdev(async_ticks)
        results['async']['variance'] = statistics.variance(async_ticks)
        print(f"  Mean: {results['async']['mean']:.2f}")
        print(f"  Std Dev: {results['async']['stdev']:.2f}")
        print(f"  Variance: {results['async']['variance']:.2f}")
    
    print(f"\nSynchronous Mode:")
    print(f"  Successful runs: {results['sync']['success_runs']}/{num_runs}")
    print(f"  Collisions: {results['sync']['num_collisions']}")
    if len(sync_ticks) > 0:
        print(f"  Collision ticks: {sync_ticks}")
    if len(sync_ticks) > 1:
        results['sync']['mean'] = statistics.mean(sync_ticks)
        results['sync']['stdev'] = statistics.stdev(sync_ticks)
        results['sync']['variance'] = statistics.variance(sync_ticks)
        print(f"  Mean: {results['sync']['mean']:.2f}")
        print(f"  Std Dev: {results['sync']['stdev']:.2f}")
        print(f"  Variance: {results['sync']['variance']:.2f}")
    
    # Compare variances
    print(f"\nVariance Comparison:")
    if len(async_ticks) > 1 and len(sync_ticks) > 1:
        variance_ratio = results['async']['variance'] / results['sync']['variance']
        results['variance_ratio'] = variance_ratio
        print(f"  Variance Ratio (Async/Sync): {variance_ratio:.3f}")
        
        if variance_ratio > 1.5:
            print(f"  → Asynchronous mode shows SIGNIFICANTLY higher variance ({variance_ratio:.2f}x)")
        elif variance_ratio > 1.1:
            print(f"  → Asynchronous mode shows moderately higher variance ({variance_ratio:.2f}x)")
        elif variance_ratio < 0.67:
            print(f"  → Synchronous mode shows SIGNIFICANTLY higher variance ({1/variance_ratio:.2f}x)")
        elif variance_ratio < 0.9:
            print(f"  → Synchronous mode shows moderately higher variance ({1/variance_ratio:.2f}x)")
        else:
            print(f"  → Variances are similar (ratio ≈ 1)")
    else:
        print(f"  Insufficient data for variance comparison")
        print(f"  (Need at least 2 collision measurements for each mode)")
    
    # Save results to JSON
    results_file = './sync_comparison/comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Compare collision tick variance between sync and async CARLA modes')
    parser.add_argument('--num_runs', type=int, default=2,
                        help='Number of simulation runs per mode (default: 10)')
    parser.add_argument('--config_type', type=str, 
                        choices=['right_turn', 'left_turn', 'merge'],
                        default='merge',
                        help='Scenario configuration type (default: merge)')
    
    args = parser.parse_args()
    
    try:
        results = run_comparison(args.num_runs, args.config_type)
        
        # Print final conclusion
        print("\n" + "=" * 60)
        print("CONCLUSION")
        print("=" * 60)
        if 'variance_ratio' in results:
            if results['variance_ratio'] > 1.5:
                print("Synchronous mode provides MORE CONSISTENT collision timing")
                print("Recommended for reproducible experiments")
            elif results['variance_ratio'] < 0.67:
                print("Asynchronous mode provides MORE CONSISTENT collision timing") 
                print("This is unexpected - please verify simulation settings")
            else:
                print("Both modes show similar consistency in collision timing")
        else:
            print("Unable to draw conclusions - more data needed")
            
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
    except Exception as e:
        print(f"\n\nError during comparison: {e}")
        import traceback
        traceback.print_exc()