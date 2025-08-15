#!/usr/bin/env python3
"""
Extensible Parking Experiment with Cloud Computing Scenarios

This experiment runner supports different cloud computing architectures:
1. Cloud Perception: Perception on cloud (delayed), planning and control local
2. Cloud Planning: Perception local, planning on cloud (delayed), control local  
3. Full Cloud: All processing on cloud (all components delayed)

Uses FogSim to accurately simulate network delays for cloud components.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from contextlib import contextmanager
import time
import subprocess
import socket
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import json
from pathlib import Path

from fogsim import FogSim, SimulationMode, NetworkConfig
from fogsim.handlers import BaseHandler

# Import extensible components
from cloud_components import CLOUD_SCENARIOS, CloudArchitectureConfig
from extensible_parking_handler import ExtensibleParkingHandler

# Import parking-specific utilities
from experiment_utils import DELTA_SECONDS

# Configure logging for CARLA management
carla_logger = logging.getLogger('carla_manager')
carla_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
carla_logger.addHandler(handler)


@dataclass
class ExtensibleScenarioConfig:
    """Configuration for extensible parking scenarios."""
    name: str
    network_delay: float = 0.0  # seconds
    packet_loss_rate: float = 0.0  # 0% loss
    source_rate: float = 1e6  # bps
    timestep: float = DELTA_SECONDS
    num_random_cars: int = 25
    replan_interval: int = 10  # Frames between replanning
    distance_threshold: float = 10.0  # Distance to trigger replanning  
    max_episode_steps: int = 5000  # Even longer for better parking
    video_fps: int = 30
    traffic_cone_positions: List[Tuple[int, int]] = None
    walker_positions: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.traffic_cone_positions is None:
            # self.traffic_cone_positions = [(284, -230), (287, -225)]
            self.traffic_cone_positions  = []
        if self.walker_positions is None:
            self.walker_positions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExtensibleScenarioConfig':
        return cls(**config_dict)


# Predefined network scenarios  
NETWORK_SCENARIOS = {
    "no_latency": ExtensibleScenarioConfig(
        name="no_latency",
        network_delay=0.0,  # 0ms
        packet_loss_rate=0.0,
        source_rate=1e9,  # High bandwidth
    ),
    
    "low_latency": ExtensibleScenarioConfig(
        name="low_latency",
        network_delay=0.005,  # 5ms
        packet_loss_rate=0,  # 0.1% loss
        source_rate=1e6,  # 1 Mbps
    ),
    
    "medium_latency": ExtensibleScenarioConfig(
        name="medium_latency", 
        network_delay=0.020,  # 20ms
        packet_loss_rate=0,  # 1% loss
        source_rate=500e3,  # 500 Kbps
    ),
    
    "high_latency": ExtensibleScenarioConfig(
        name="high_latency",
        network_delay=0.050,  # 50ms
        packet_loss_rate=0,  # 3% loss
        source_rate=200e3,  # 200 Kbps
    ),
}


def run_extensible_parking_scenario(mode: SimulationMode,
                                   scenario_config: ExtensibleScenarioConfig,
                                   cloud_config: CloudArchitectureConfig,
                                   destination: int,
                                   parked_spots: List[int],
                                   recording_file=None,
                                   verbose: bool = True) -> Dict:
    """Run a single parking scenario with extensible cloud architecture."""
    
    if verbose:
        print(f'  Scenario: {cloud_config.name} - parking spot {destination}, occupied: {parked_spots}')
        
    # Create extensible parking handler
    handler = ExtensibleParkingHandler(scenario_config, cloud_config)
    handler.set_scenario(destination, parked_spots)
    if recording_file:
        handler.set_recording(recording_file)
    
    # Configure network for cloud delays
    network_config = NetworkConfig()
    network_config.topology.link_delay = scenario_config.network_delay
    network_config.topology.link_bandwidth = scenario_config.source_rate * 8.0  # Convert to bits/sec
    network_config.source_rate = scenario_config.source_rate
    network_config.packet_loss_rate = scenario_config.packet_loss_rate
    
    # Create FogSim environment
    fogsim = FogSim(
        handler, 
        mode=mode, 
        timestep=scenario_config.timestep,
        network_config=network_config
    )
    
    # Reset environment
    obs, info = fogsim.reset()
    
    # Run episode
    step_count = 0
    total_latency = 0.0
    latency_samples = 0
    cloud_messages_sent = 0
    cloud_messages_received = 0
    
    while not handler._episode_done and step_count < scenario_config.max_episode_steps:
        # For cloud scenarios, send observation data through network
        # For baseline (all local), send None
        if cloud_config.name != 'baseline' and step_count % scenario_config.replan_interval == 0:
            # Send observation through network for cloud processing
            action_to_send = obs
            cloud_messages_sent += 1  # Count actual message sends
            if verbose and step_count % 50 == 0:
                print(f"    Sending observation to cloud at step {step_count}")
        else:
            action_to_send = None
            
        # Step through FogSim
        result = fogsim.step(action_to_send)
        
        if isinstance(result, tuple) and len(result) >= 6:
            obs, reward, success, terminated, truncated, info = result
        else:
            # Handle different return formats
            obs, reward, success, terminated, truncated, info = result, 0, False, False, False, {}
        
        step_count += 1
        
        # Track cloud message statistics  
        if info.get('actions_received', 0) > 0:
            cloud_messages_received += info.get('actions_received', 0)
        if info.get('observations_received', 0) > 0:
            cloud_messages_received += info.get('observations_received', 0)
        
        # Check if episode should end
        if truncated or terminated:
            break
            
        # Progress reporting
        if verbose and step_count % 50 == 0:
            cloud_info = f"Cloud: {cloud_messages_sent} sent, {cloud_messages_received} recv"
            if 'avg_latency_ms' in info:
                avg_latency = info['avg_latency_ms']
                total_latency += avg_latency
                latency_samples += 1
                print(f'    Step {step_count}: {cloud_info}, latency={avg_latency:.1f}ms')
            else:
                print(f'    Step {step_count}: {cloud_info}, no latency measured yet')
    
    # Get final results
    final_iou = handler._last_iou
    final_parking_time = handler._parking_time
    
    if verbose:
        parking_time_str = f", Parking Time: {final_parking_time:.2f}s" if final_parking_time else ""
        cloud_summary = f"Cloud msgs: {cloud_messages_sent}→{cloud_messages_received}"
        
        if latency_samples > 0:
            avg_total_latency = total_latency / latency_samples
            print(f'  Result: IOU={final_iou:.3f}{parking_time_str}, {cloud_summary}, Avg Latency={avg_total_latency:.1f}ms')
        else:
            print(f'  Result: IOU={final_iou:.3f}{parking_time_str}, {cloud_summary}')
    
    # Clean up
    fogsim.close()
    
    return {
        'mode': mode,
        'cloud_config': cloud_config.name,
        'destination': destination,
        'parked_spots': parked_spots,
        'iou': final_iou,
        'parking_time': final_parking_time,
        'steps': handler.frame_idx,
        'cloud_messages_sent': cloud_messages_sent,
        'cloud_messages_received': cloud_messages_received,
        'avg_latency_ms': total_latency / latency_samples if latency_samples > 0 else 0,
    }


def run_cloud_scenario_experiments(mode: SimulationMode,
                                  scenario_config: ExtensibleScenarioConfig,
                                  cloud_configs: List[CloudArchitectureConfig],
                                  scenarios: List[Tuple[int, List[int]]],
                                  num_trials: int = 3,
                                  video_enabled: bool = False,
                                  verbose: bool = True) -> Dict[str, Tuple[List[float], List[Optional[float]]]]:
    """Run experiments for multiple cloud configurations."""
    
    results = {}
    
    for cloud_config in cloud_configs:
        if verbose:
            print(f'\n=== Testing {cloud_config.name}: {cloud_config.description} ===')
        
        ious = []
        parking_times = []
        
        for idx, (destination, parked_spots) in enumerate(scenarios):
            if verbose:
                print(f'    Scenario {idx+1}: Destination {destination}, Occupied spots {parked_spots}')
            
            # Run multiple trials for this scenario
            scenario_ious = []
            scenario_times = []
            
            for trial_num in range(num_trials):
                if verbose:
                    print(f'      Trial {trial_num+1}/{num_trials}')
                
                recording_file = None
                
                # Create unique video filename if video is enabled
                if video_enabled:
                    video_filename = f'parking_{cloud_config.name}_latency{int(scenario_config.network_delay*1000)}ms_rate{int(scenario_config.source_rate/1000)}kbps_dest{destination}_scenario{idx+1}_trial{trial_num+1}.mp4'
                    recording_file = iio.imopen(video_filename, 'w', plugin='pyav')
                    recording_file.init_video_stream('vp9', fps=scenario_config.video_fps)
                    if verbose:
                        print(f'        Recording video to: {video_filename}')
                
                result = run_extensible_parking_scenario(
                    mode=mode,
                    scenario_config=scenario_config,
                    cloud_config=cloud_config,
                    destination=destination,
                    parked_spots=parked_spots,
                    recording_file=recording_file,
                    verbose=verbose
                )
                
                scenario_ious.append(result['iou'])
                scenario_times.append(result.get('parking_time'))
                
                # Close recording file
                if recording_file:
                    recording_file.close()
            
            # Add all trial results to the overall lists
            ious.extend(scenario_ious)
            parking_times.extend(scenario_times)
            
            # Print trial summary for this scenario
            if verbose and scenario_ious:
                print(f'      Scenario summary: IOU = {np.mean(scenario_ious):.3f} ± {np.std(scenario_ious):.3f}')
        
        results[cloud_config.name] = (ious, parking_times)
        
        # Print summary for this cloud configuration
        if ious and verbose:
            total_trials = len(ious)
            print(f'  Summary ({total_trials} trials): mean IOU = {np.mean(ious):.3f}, std = {np.std(ious):.3f}')
            valid_times = [t for t in parking_times if t is not None]
            if valid_times:
                print(f'           mean parking time = {np.mean(valid_times):.2f}s, std = {np.std(valid_times):.2f}s')
    
    return results


def plot_cloud_comparison_results(cloud_results: Dict[str, Tuple[List[float], List[Optional[float]]]], 
                                 scenario_name: str,
                                 latency_ms: float,
                                 save_path: str = None,
                                 num_trials: int = 1):
    """Plot comparison results across different cloud architectures."""
    
    if save_path is None:
        save_path = f'parking_cloud_comparison_{scenario_name}_latency{int(latency_ms)}ms.png'
    
    # Prepare JSON data to save alongside the plot
    json_data = {
        'scenario_name': scenario_name,
        'latency_ms': latency_ms,
        'num_trials': num_trials,
        'results': {}
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Colors for different cloud scenarios
    colors = {
        'baseline': 'blue',
        'cloud_perception': 'orange', 
        'cloud_planning': 'green',
        'full_cloud': 'red'
    }
    
    cloud_names = []
    iou_means = []
    iou_stds = []
    time_means = []
    time_stds = []
    
    # Plot 1: IOU comparison
    for cloud_name, (ious, parking_times) in cloud_results.items():
        if ious:
            cloud_names.append(cloud_name.replace('_', '\n'))
            iou_means.append(np.mean(ious))
            iou_stds.append(np.std(ious))
            
            # Store data for JSON export
            json_data['results'][cloud_name] = {
                'ious': ious,
                'parking_times': parking_times,
                'iou_mean': float(np.mean(ious)),
                'iou_std': float(np.std(ious)),
                'iou_min': float(np.min(ious)),
                'iou_max': float(np.max(ious))
            }
            
            # Add parking time statistics if available
            valid_times = [t for t in parking_times if t is not None]
            if valid_times:
                json_data['results'][cloud_name].update({
                    'parking_time_mean': float(np.mean(valid_times)),
                    'parking_time_std': float(np.std(valid_times)),
                    'parking_time_min': float(np.min(valid_times)),
                    'parking_time_max': float(np.max(valid_times)),
                    'valid_parking_times_count': len(valid_times)
                })
            
            # Scatter plot of individual results
            x_positions = np.random.normal(len(cloud_names)-1, 0.1, len(ious))
            ax1.scatter(x_positions, ious, alpha=0.6, 
                       color=colors.get(cloud_name, 'gray'), s=50)
    
    # Bar plot with error bars for IOU
    bars1 = ax1.bar(range(len(cloud_names)), iou_means, yerr=iou_stds, 
                    capsize=5, alpha=0.7, 
                    color=[colors.get(name.replace('\n', '_'), 'gray') for name in cloud_names])
    
    ax1.set_xlabel('Cloud Architecture')
    ax1.set_ylabel('Parking IOU')
    ax1.set_title(f'Parking Performance by Cloud Architecture\n({scenario_name}, {latency_ms}ms latency, {num_trials} trials)')
    ax1.set_xticks(range(len(cloud_names)))
    ax1.set_xticklabels(cloud_names)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Parking time comparison
    for i, (cloud_name, (ious, parking_times)) in enumerate(cloud_results.items()):
        valid_times = [t for t in parking_times if t is not None]
        if valid_times:
            time_means.append(np.mean(valid_times))
            time_stds.append(np.std(valid_times))
            
            # Scatter plot of individual results
            x_positions = np.random.normal(i, 0.1, len(valid_times))
            ax2.scatter(x_positions, valid_times, alpha=0.6,
                       color=colors.get(cloud_name, 'gray'), s=50)
        else:
            time_means.append(0)
            time_stds.append(0)
    
    # Bar plot with error bars for parking time
    bars2 = ax2.bar(range(len(cloud_names)), time_means, yerr=time_stds,
                    capsize=5, alpha=0.7,
                    color=[colors.get(name.replace('\n', '_'), 'gray') for name in cloud_names])
    
    ax2.set_xlabel('Cloud Architecture')
    ax2.set_ylabel('Parking Time (seconds)')
    ax2.set_title(f'Parking Time by Cloud Architecture\n({scenario_name}, {latency_ms}ms latency, {num_trials} trials)')
    ax2.set_xticks(range(len(cloud_names)))
    ax2.set_xticklabels(cloud_names)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Cloud comparison results saved to {save_path}')
    
    # Save JSON data alongside the plot
    json_save_path = save_path.replace('.png', '_data.json')
    with open(json_save_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f'Raw data saved to {json_save_path}')


def main():
    """Main experiment runner for extensible cloud parking scenarios."""
    parser = argparse.ArgumentParser(description="Extensible Cloud Parking Experiment")
    parser.add_argument("--modes", nargs='+', choices=['virtual', 'simulated', 'real'],
                       default=['virtual'], 
                       help="FogSim modes to test")
    parser.add_argument("--clouds", nargs='+', 
                       choices=list(CLOUD_SCENARIOS.keys()),
                       default=['cloud_perception', 'cloud_planning'],
                       help="Cloud scenarios to test")
    parser.add_argument("--latency", type=float, nargs='+', default=[10, 50, 75, 100, 150, 300],
                       help="List of network latencies in ms to test")
    parser.add_argument("--bandwidth", type=float, nargs='+', default=[100000000.0],
                       help="List of network bandwidths in kbps to test")
    parser.add_argument("--video", action="store_true", 
                       help="Enable video recording")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip visualization plotting")
    parser.add_argument("--config", type=str, 
                       help="Path to custom network config JSON file")
    parser.add_argument("--save-config", type=str, 
                       help="Save current config to JSON file")
    parser.add_argument("--trials", type=int, default=3,
                       help="Number of trials to run for each configuration")
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        base_scenario_config = ExtensibleScenarioConfig.from_dict(config_dict)
    else:
        # Use default base configuration
        base_scenario_config = ExtensibleScenarioConfig(
            name="custom",
            network_delay=0.005,  # 5ms default
            packet_loss_rate=0.0,
            source_rate=1e6,  # 1000 kbps default
        )
    
    # Get latency and bandwidth values to test
    latency_values = args.latency
    bandwidth_values = args.bandwidth
        
    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(base_scenario_config.to_dict(), f, indent=2)
        print(f"Configuration saved to {args.save_config}")
    
    # Get cloud configurations to test
    cloud_configs = [CLOUD_SCENARIOS[name] for name in args.clouds]
    
    # Define parking scenarios - test just one for now
    SCENARIOS = [
        (20, [19, 21]),
    ]
    
    try:
        # Map mode names to SimulationMode enum
        mode_map = {
            'virtual': SimulationMode.VIRTUAL,
            'simulated': SimulationMode.SIMULATED_NET,
            'real': SimulationMode.REAL_NET
        }
        
        all_results = {}
        
        for mode_name in args.modes:
            mode = mode_map[mode_name]
            mode_results = {}
            
            for latency_ms in latency_values:
                for bandwidth_kbps in bandwidth_values:
                    # Calculate appropriate timestep for this latency
                    # FIXED: Timestep should be SMALLER than network delay for better granularity
                    # This allows multiple simulation steps within the delay period for proper message handling
                    network_delay_s = latency_ms / 1000.0  # Convert ms to seconds
                    
                    timestep = DELTA_SECONDS
                    # if network_delay_s <= 0.010:  # <= 10ms delay
                    #     timestep = 0.005  # 5ms timestep (2x finer than delay)
                    # elif network_delay_s <= 0.025:  # <= 25ms delay  
                    #     timestep = 0.01   # 10ms timestep (2.5x finer than delay)
                    # elif network_delay_s <= 0.060:  # <= 60ms delay
                    #     timestep = 0.02   # 20ms timestep (3x finer than delay)
                    # elif network_delay_s <= 0.120:  # <= 120ms delay
                    #     timestep = 0.03   # 30ms timestep (4x finer than delay)
                    # else:  # > 120ms delay
                    #     timestep = 0.05   # 50ms timestep (still reasonable granularity)
                    
                    print(f"    Using timestep: {timestep*1000:.0f}ms for {latency_ms}ms network delay")
                    
                    # Create a copy of scenario config with current network parameters
                    current_scenario_config = ExtensibleScenarioConfig(
                        name=f"custom_latency{int(latency_ms)}ms_bw{int(bandwidth_kbps)}kbps",
                        network_delay=network_delay_s,
                        packet_loss_rate=base_scenario_config.packet_loss_rate,
                        source_rate=bandwidth_kbps * 1000.0,  # Convert kbps to bps
                        timestep=timestep,  # Use calculated timestep
                        num_random_cars=base_scenario_config.num_random_cars,
                        replan_interval=base_scenario_config.replan_interval,
                        distance_threshold=base_scenario_config.distance_threshold,
                        max_episode_steps=int(base_scenario_config.max_episode_steps * (DELTA_SECONDS/timestep)),  # Adjust for timestep
                        video_fps=base_scenario_config.video_fps,
                        traffic_cone_positions=base_scenario_config.traffic_cone_positions,
                        walker_positions=base_scenario_config.walker_positions
                    )
                    
                    print(f'\n{"="*80}')
                    print(f'TESTING MODE: {mode_name.upper()} - LATENCY: {latency_ms}ms, BANDWIDTH: {bandwidth_kbps}kbps')
                    print(f'Network: {current_scenario_config.name}')
                    print(f'Cloud scenarios: {[cfg.name for cfg in cloud_configs]}')
                    print(f'{"="*80}')
                    
                    # Run experiments for all cloud scenarios with current network config
                    network_results = run_cloud_scenario_experiments(
                        mode=mode,
                        scenario_config=current_scenario_config,
                        cloud_configs=cloud_configs,
                        scenarios=SCENARIOS,
                        num_trials=args.trials,
                        video_enabled=args.video,
                        verbose=True
                    )
                    
                    # Store results with network parameters key
                    for cloud_name, results in network_results.items():
                        key = f"{cloud_name}_latency{int(latency_ms)}ms_bw{int(bandwidth_kbps)}kbps"
                        mode_results[key] = results
                    
                    # Generate visualization for this mode and network configuration
                    if not args.no_plot:
                        plot_cloud_comparison_results(
                            network_results, 
                            f"{mode_name}_{current_scenario_config.name}",
                            current_scenario_config.network_delay * 1000,
                            f'parking_cloud_{mode_name}_latency{int(latency_ms)}ms_bw{int(bandwidth_kbps)}kbps.png',
                            num_trials=args.trials
                        )
            
            all_results[mode_name] = mode_results
        
        # Print final comparison summary
        print(f"\n{'='*80}")
        print(f"FINAL CLOUD ARCHITECTURE COMPARISON ({args.trials} trials per scenario)")
        print(f"{'='*80}")
        
        for mode_name, mode_results in all_results.items():
            print(f"\n{mode_name.upper()} MODE RESULTS:")
            print("-" * 50)
            
            for result_key, (ious, parking_times) in mode_results.items():
                # Extract base cloud name from result key like "cloud_perception_latency100ms_bw100000000kbps"
                base_cloud_name = result_key.split('_latency')[0]
                cloud_config = CLOUD_SCENARIOS[base_cloud_name]
                if ious:
                    total_trials = len(ious)
                    print(f"\n{result_key.upper()}: {cloud_config.description}")
                    print(f"  IOU ({total_trials} trials): {np.mean(ious):.3f} ± {np.std(ious):.3f} (range: {np.min(ious):.3f}-{np.max(ious):.3f})")
                    
                    valid_times = [t for t in parking_times if t is not None]
                    if valid_times:
                        print(f"  Time ({len(valid_times)} trials): {np.mean(valid_times):.2f}s ± {np.std(valid_times):.2f}s")
        
        # Compare cloud scenarios within each mode
        for mode_name, mode_results in all_results.items():
            if len(mode_results) > 1:
                print(f"\n{mode_name.upper()} MODE - Cloud Architecture Impact:")
                # Find baseline results (any key starting with 'baseline')
                baseline_ious = []
                for key, (ious, _) in mode_results.items():
                    if key.startswith('baseline'):
                        baseline_ious = ious
                        break
                if baseline_ious:
                    baseline_mean = np.mean(baseline_ious)
                    print(f"  Baseline (all local): {baseline_mean:.3f}")
                    
                    for result_key, (ious, _) in mode_results.items():
                        if not result_key.startswith('baseline') and ious:
                            cloud_mean = np.mean(ious)
                            impact = cloud_mean - baseline_mean
                            print(f"  {result_key}: {cloud_mean:.3f} ({impact:+.3f} vs baseline)")
        
        # Save all results to JSON file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f'parking_experiment_results_{timestamp}.json'
        
        # Convert all_results to JSON-serializable format
        json_results = {}
        for mode_name, mode_results in all_results.items():
            json_results[mode_name] = {}
            for result_key, (ious, parking_times) in mode_results.items():
                # Convert None values to null for JSON compatibility
                clean_parking_times = [t if t is not None else None for t in parking_times]
                
                json_results[mode_name][result_key] = {
                    'ious': ious,
                    'parking_times': clean_parking_times,
                    'statistics': {
                        'iou_mean': float(np.mean(ious)) if ious else None,
                        'iou_std': float(np.std(ious)) if ious else None,
                        'iou_min': float(np.min(ious)) if ious else None,
                        'iou_max': float(np.max(ious)) if ious else None,
                        'trial_count': len(ious)
                    }
                }
                
                # Add parking time statistics
                valid_times = [t for t in parking_times if t is not None]
                if valid_times:
                    json_results[mode_name][result_key]['statistics'].update({
                        'parking_time_mean': float(np.mean(valid_times)),
                        'parking_time_std': float(np.std(valid_times)),
                        'parking_time_min': float(np.min(valid_times)),
                        'parking_time_max': float(np.max(valid_times)),
                        'valid_parking_times_count': len(valid_times)
                    })
        
        # Add experiment metadata
        experiment_metadata = {
            'timestamp': timestamp,
            'experiment_config': {
                'modes': args.modes,
                'clouds': args.clouds,
                'latency_values': latency_values,
                'bandwidth_values': bandwidth_values,
                'trials': args.trials,
                'video_enabled': args.video,
                'scenarios': SCENARIOS
            },
            'results': json_results
        }
        
        with open(json_filename, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        print(f"\nAll experiment results saved to {json_filename}")
        
    except KeyboardInterrupt:
        print('\nExperiment interrupted by user')
    except Exception as e:
        print(f'Error during experiment: {e}')
        raise


if __name__ == '__main__':
    main()