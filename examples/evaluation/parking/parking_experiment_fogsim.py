#!/usr/bin/env python3
"""
Parking Experiment with FogSim Network Simulation Modes

This version faithfully implements the original parking_experiment.py logic
while using FogSim's three simulation modes to compare parking performance 
under different network conditions:
1. Virtual Timeline - Decoupled from wallclock for fast simulation  
2. Real Clock + Simulated Network - Network delays simulated with NS-3
3. Real Clock + Real Network - Actual network latency

Demonstrates FogSim's capability to handle latency in autonomous parking scenarios.
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio
from contextlib import contextmanager
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import argparse
import json
from pathlib import Path

from fogsim import FogSim, SimulationMode, NetworkConfig
from fogsim.handlers import BaseHandler

# Import parking-specific utilities
from experiment_utils import (
    load_client,
    is_done,
    town04_load,
    town04_spectator_bev,
    town04_spawn_ego_vehicle,
    town04_spawn_parked_cars,
    town04_spawn_traffic_cones,
    town04_spawn_walkers,
    update_walkers,
    obstacle_map_from_bbs,
    clear_obstacle_map,
    union_obstacle_map,
    mask_obstacle_map,
    DELTA_SECONDS
)


@dataclass
class ParkingScenarioConfig:
    """Configuration for parking scenarios matching original experiment."""
    name: str
    network_delay: float = 0.0  # seconds (0ms default like original)
    packet_loss_rate: float = 0.0  # 0% loss
    source_rate: float = 1e6  # bps
    timestep: float = DELTA_SECONDS
    num_random_cars: int = 25
    replan_interval: int = 10  # Frames between replanning
    distance_threshold: float = 10.0  # Distance to trigger replanning  
    max_episode_steps: int = 10000  # Large limit
    video_fps: int = 30
    traffic_cone_positions: List[Tuple[int, int]] = None
    walker_positions: List[Tuple[int, int]] = None
    
    def __post_init__(self):
        if self.traffic_cone_positions is None:
            self.traffic_cone_positions = [(284, -230), (287, -225)]
        if self.walker_positions is None:
            self.walker_positions = []
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ParkingScenarioConfig':
        return cls(**config_dict)


# Predefined scenarios with different network conditions
PREDEFINED_SCENARIOS = {
    "no_latency": ParkingScenarioConfig(
        name="no_latency",
        network_delay=0.0,  # 0ms (like original)
        packet_loss_rate=0.0,  # 0% loss
    ),
    
    "low_latency": ParkingScenarioConfig(
        name="low_latency",
        network_delay=0.05,  # 50ms
        packet_loss_rate=0.001,  # 0.1% loss
    ),
    
    "medium_latency": ParkingScenarioConfig(
        name="medium_latency", 
        network_delay=0.15,  # 150ms
        packet_loss_rate=0.01,  # 1% loss
    ),
    
    "high_latency": ParkingScenarioConfig(
        name="high_latency",
        network_delay=0.3,  # 300ms
        packet_loss_rate=0.03,  # 3% loss
    ),
}


class ParkingHandler(BaseHandler):
    """Handler for parking scenario in CARLA, matching original logic.
    
    This handler encapsulates the parking logic from the original experiment
    and interfaces with FogSim to handle network delays through different modes.
    """
    
    def __init__(self, config: ParkingScenarioConfig):
        """Initialize the parking handler."""
        self.config = config
        self.client = None
        self.world = None
        self.car = None
        self.recording_cam = None
        self.recording_file = None
        self.actors_to_cleanup = []
        self.static_bbs = []
        self.dynamic_bbs = []
        self.walkers = []
        self.destination_parking_spot = None
        self.parked_spots = []
        self.frame_idx = 0
        self._launched = False
        self._observation = None
        self._last_iou = 0.0
        self._episode_done = False
        # Perception state for latency handling (matches original)
        self.perception_state = {'request': None, 'response': None}
        
    def launch(self) -> None:
        """Launch CARLA and initialize the parking scenario."""
        if self._launched:
            return
            
        # Initialize CARLA
        self.client = load_client()
        self.world = town04_load(self.client)
        town04_spectator_bev(self.world)
        
        self._launched = True
        
    def set_scenario(self, destination: int, parked_spots: List[int]):
        """Set the parking scenario parameters."""
        self.destination_parking_spot = destination
        self.parked_spots = parked_spots
    
    def set_recording(self, recording_file):
        """Set video recording file."""
        self.recording_file = recording_file
        
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the parking scenario and return initial observation."""
        self.set_states(None)  # Trigger reset
        states = self.get_states()
        return states['observation'], states
        
    def _spawn_actors(self):
        """Spawn all actors in the scenario."""
        # Spawn parked cars
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(
            self.world, self.parked_spots, 
            self.destination_parking_spot, self.config.num_random_cars
        )
        self.actors_to_cleanup.extend(parked_cars)
        
        # Spawn traffic cones
        traffic_cones, traffic_cone_bbs = town04_spawn_traffic_cones(
            self.world, self.config.traffic_cone_positions
        )
        self.actors_to_cleanup.extend(traffic_cones)
        
        # Spawn walkers
        walkers, walker_bbs = town04_spawn_walkers(
            self.world, self.config.walker_positions
        )
        self.actors_to_cleanup.extend(walkers)
        self.walkers = walkers
        
        # Store static obstacles
        self.static_bbs = parked_cars_bbs + traffic_cone_bbs
        
        self.world.tick()  # Load actors
        
    def _cleanup_actors(self):
        """Clean up all spawned actors."""
        for actor in self.actors_to_cleanup:
            if actor is not None:
                actor.destroy()
        self.actors_to_cleanup.clear()
        
    def set_states(self, states: Optional[Dict[str, Any]] = None,
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states (used for reset and action application).
        
        Args:
            states: Reset signal (None triggers reset)
            action: Not used in this version (actions handled in step)
        """
        if states is None:
            # Reset scenario
            self._reset_scenario()
            
    def _reset_scenario(self):
        """Reset the parking scenario matching original logic."""
        # Clean up previous actors
        if self.recording_cam is not None:
            self.recording_cam.destroy()
            self.recording_cam = None
        if self.car is not None:
            self.car.destroy()
            self.car = None
        self._cleanup_actors()
        
        # Spawn new actors
        self._spawn_actors()
        
        # Initialize ego vehicle
        self.car = town04_spawn_ego_vehicle(
            self.world, self.destination_parking_spot
        )
        
        # Initialize video recording if configured
        if self.recording_file is not None:
            self.recording_cam = self.car.init_recording(self.recording_file)
        
        # Initialize perception with static obstacles (matches original)
        self.car.car.obs = clear_obstacle_map(
            obstacle_map_from_bbs(self.static_bbs)
        )
        
        # Save obstacle map visualization (matches original)
        self._save_obstacle_map(self.car.car.obs.obs)
        
        # Reset perception state
        self.perception_state = {'request': None, 'response': None}
        
        # Reset state variables
        self.frame_idx = 0
        self._episode_done = False
        self._last_iou = 0.0
        
        # No initial world.tick() or car.localize() or car.plan() here
        # They will be done in the main loop like the original
        
        # Create initial observation
        self._update_observation()
    
    def _save_obstacle_map(self, obs_map):
        """Save obstacle map visualization (matches original)."""
        plt.figure(figsize=(8, 8))
        plt.imshow(obs_map, cmap='gray')
        plt.title('Obstacle Map')
        plt.axis('off')
        plt.savefig('obs_map.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    def _update_perception(self):
        """Handle perception updates based on latency (matches original logic)."""
        latency_ms = self.config.network_delay * 1000  # Convert to ms
        
        if latency_ms == 0:
            # No latency - immediate perception (matches original)
            walker_bbs = update_walkers(self.walkers) if self.walkers else []
            all_bbs = self.static_bbs + walker_bbs
            self.car.car.obs = union_obstacle_map(
                self.car.car.obs,
                mask_obstacle_map(
                    obstacle_map_from_bbs(all_bbs),
                    self.car.car.cur.x,
                    self.car.car.cur.y
                )
            )
        elif latency_ms > 0 and self.frame_idx % int(latency_ms / 1000 / DELTA_SECONDS) == 0:
            # Latency-based perception (matches original)
            self.car.perceive()
            
            if self.perception_state['response']:
                self.car.car.obs = self.perception_state['response']
            
            if self.perception_state['request']:
                walker_bbs = update_walkers(self.walkers) if self.walkers else []
                all_bbs = self.static_bbs + walker_bbs
                self.perception_state['response'] = union_obstacle_map(
                    self.car.car.obs,
                    mask_obstacle_map(
                        obstacle_map_from_bbs(all_bbs),
                        self.perception_state['request'].x,
                        self.perception_state['request'].y
                    )
                )
            
            self.perception_state['request'] = self.car.car.cur
            
    def _should_replan(self):
        """Determine if replanning is needed (matches original)."""
        return (self.frame_idx % self.config.replan_interval == 0 and 
                self.car.car.cur.distance(self.car.car.destination) > self.config.distance_threshold)
        
    def _update_observation(self):
        """Update the observation vector."""
        if self.car is None:
            self._observation = np.zeros(15)
            return
            
        # Get car state
        transform = self.car.actor.get_transform()
        velocity = self.car.actor.get_velocity()
        
        # Calculate distance to destination
        dist_to_dest = self.car.car.cur.distance(self.car.car.destination)
        
        # Check if done
        self._episode_done = is_done(self.car)
        
        # Calculate IoU if parked
        if self._episode_done:
            self._last_iou = self.car.iou()
            
        # Create observation vector
        self._observation = np.array([
            transform.location.x,
            transform.location.y,
            transform.rotation.yaw,
            velocity.x,
            velocity.y,
            dist_to_dest,
            float(self._episode_done),
            self._last_iou,
            self.frame_idx / self.config.max_episode_steps,
            # Obstacle map features (simplified)
            np.sum(self.car.car.obs.obs) / (self.car.car.obs.obs.size + 1e-6),
            np.mean(self.car.car.obs.obs),
            np.std(self.car.car.obs.obs),
            # Parking spot location
            self.car.car.destination.x,
            self.car.car.destination.y,
            self.car.car.destination.angle,
        ])
        
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states.
        
        Returns:
            Dictionary with observation and metadata
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
            
        self._update_observation()
        
        return {
            'observation': self._observation,
            'done': self._episode_done,
            'iou': self._last_iou,
            'frame': self.frame_idx,
            'car_position': [self.car.car.cur.x, self.car.car.cur.y] if self.car else [0, 0],
            'destination': [self.car.car.destination.x, self.car.car.destination.y] if self.car else [0, 0],
        }
        
    def step(self) -> None:
        """Step the simulation forward matching original logic."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self._episode_done:
            return
            
        # Update dynamic obstacles (walkers)
        if self.walkers:
            self.dynamic_bbs = update_walkers(self.walkers)
        
        # Tick simulation
        self.world.tick()
        self.car.localize()
        
        # Update perception (matches original update_perception logic)
        self._update_perception()
        
        # Replan if needed (matches original should_replan)
        if self._should_replan():
            self.car.plan()
        
        # Execute step
        self.car.run_step()
        
        # Process recording frames if video recording is enabled
        if self.recording_file:
            latency_ms = self.config.network_delay * 1000
            self.car.process_recording_frames(latency=latency_ms)
        
        # Increment frame counter
        self.frame_idx += 1
        
    def step_with_action(self, action: Optional[np.ndarray]) -> Tuple:
        """Step environment with action and return results (required by FogSim).
        
        Returns:
            Tuple of (observation, reward, success, terminated, truncated, info)
        """
        # Execute step (action is ignored, logic is in step())
        self.step()
        
        # Get new state
        states = self.get_states()
        
        # Calculate simple reward
        reward = 0.0
        if states['done']:
            reward = states['iou'] * 100  # Bonus for good parking
            
        # Determine termination
        terminated = states['done']
        truncated = self.frame_idx >= self.config.max_episode_steps
        success = states['iou'] >= 0.8 if states['done'] else False
        
        return states['observation'], reward, success, terminated, truncated, states
        
    def render(self) -> Optional[np.ndarray]:
        """Render is not implemented for this handler."""
        return None
        
    def close(self) -> None:
        """Clean up resources."""
        if self.recording_cam is not None:
            self.recording_cam.destroy()
            self.recording_cam = None
        if self.car is not None:
            self.car.destroy()
            self.car = None
        self._cleanup_actors()
        if self.world is not None:
            self.world.tick()
        self._launched = False
        
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata."""
        return {
            'config': self.config.to_dict(),
            'launched': self._launched,
            'destination': self.destination_parking_spot,
            'parked_spots': self.parked_spots,
        }
        
    @property
    def is_launched(self) -> bool:
        """Check if handler is launched."""
        return self._launched


def run_parking_scenario(mode: SimulationMode,
                        config: ParkingScenarioConfig,
                        destination: int,
                        parked_spots: List[int],
                        recording_file=None,
                        verbose: bool = True) -> Dict:
    """Run a single parking scenario with FogSim matching original logic.
    
    Args:
        mode: FogSim simulation mode
        config: Parking scenario configuration
        destination: Destination parking spot ID
        parked_spots: List of occupied parking spot IDs
        recording_file: Optional video recording file
        verbose: Print progress
        
    Returns:
        Dictionary with results
    """
    if verbose:
        print(f'  Scenario: parking spot {destination}, occupied: {parked_spots}')
        
    # Create parking handler
    handler = ParkingHandler(config)
    handler.set_scenario(destination, parked_spots)
    if recording_file:
        handler.set_recording(recording_file)
    
    # Configure network
    network_config = NetworkConfig()
    network_config.topology.link_delay = config.network_delay
    network_config.source_rate = config.source_rate
    network_config.packet_loss_rate = config.packet_loss_rate
    
    # Create FogSim environment
    fogsim = FogSim(
        handler, 
        mode=mode, 
        timestep=config.timestep,
        network_config=network_config
    )
    
    # Reset environment
    obs, info = fogsim.reset()
    
    # Run episode matching original loop structure
    while not handler._episode_done:
        # Simple step without action (action logic is in handler.step())
        fogsim.step(None)
    
    # Get final IoU
    final_iou = handler._last_iou
    if verbose:
        print(f'IOU: {final_iou}')
    
    # Clean up
    fogsim.close()
    
    return {
        'mode': mode,
        'destination': destination,
        'parked_spots': parked_spots,
        'iou': final_iou,
        'steps': handler.frame_idx,
    }


def run_experiments_for_mode(mode: SimulationMode,
                            config: ParkingScenarioConfig,
                            scenarios: List[Tuple[int, List[int]]],
                            recording_file=None,
                            verbose: bool = True) -> List[float]:
    """Run all scenarios for a single mode (matches original structure)."""
    
    ious = []
    
    for destination, parked_spots in scenarios:
        result = run_parking_scenario(
            mode=mode,
            config=config,
            destination=destination,
            parked_spots=parked_spots,
            recording_file=recording_file,
            verbose=verbose
        )
        ious.append(result['iou'])
    
    return ious


def plot_comparison_results(latency_ious: List[Tuple[str, float, List[float]]], 
                           save_path: str = 'parking_fogsim_results.png'):
    """Plot comparison results across different FogSim modes and latencies."""
    
    plt.figure(figsize=(12, 8))
    
    colors = {'virtual': 'blue', 'simulated': 'orange', 'real': 'green'}
    markers = {'virtual': 'o', 'simulated': 's', 'real': '^'}
    
    for mode_name, latency_ms, ious in latency_ious:
        if len(ious) > 0:
            # Add some jitter to x-axis for visibility
            x_scatter = np.random.normal(loc=latency_ms, scale=2, size=len(ious))
            plt.scatter(x_scatter, ious, alpha=0.6, 
                       label=f'{mode_name} ({latency_ms}ms)', 
                       color=colors.get(mode_name, 'gray'),
                       marker=markers.get(mode_name, 'o'),
                       s=80)
    
    plt.title('Parking IOU Values vs Network Latency (FogSim Modes)', fontsize=14)
    plt.xlabel('Network Latency (ms)', fontsize=12)
    plt.ylabel('IOU Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'\nResults saved to {save_path}')


def main():
    """Main experiment runner matching original structure."""
    parser = argparse.ArgumentParser(description="FogSim Parking Experiment")
    parser.add_argument("--modes", nargs='+', choices=['virtual', 'simulated', 'real'],
                       default=['virtual', 'simulated', 'real'], 
                       help="Modes to compare")
    parser.add_argument("--latencies", nargs='+', type=float,
                       default=[0], 
                       help="Network latencies in ms (default: [0])")
    parser.add_argument("--no-video", action="store_true", 
                       help="Skip video recording for faster execution")
    parser.add_argument("--no-plot", action="store_true",
                       help="Skip visualization plotting")
    parser.add_argument("--scenario", type=str, default="no_latency",
                       choices=list(PREDEFINED_SCENARIOS.keys()),
                       help="Predefined network scenario")
    parser.add_argument("--config", type=str, help="Path to custom config JSON file")
    parser.add_argument("--save-config", type=str, 
                       help="Save current config to JSON file")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = ParkingScenarioConfig.from_dict(config_dict)
    else:
        config = PREDEFINED_SCENARIOS[args.scenario]
        
    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"Configuration saved to {args.save_config}")
    
    # Define parking scenarios (matches original)
    SCENARIOS = [
        (20, [19, 21]),
        (21, [20, 22]),
        (22, [21, 23]),
    ]
    
    recording_file = None
    
    try:
        # Setup video recording if not disabled
        if not args.no_video:
            recording_file = iio.imopen('./test.mp4', 'w', plugin='pyav')
            recording_file.init_video_stream('vp9', fps=config.video_fps)
        
        # Map mode names to SimulationMode enum
        mode_map = {
            'virtual': SimulationMode.VIRTUAL,
            'simulated': SimulationMode.SIMULATED_NET,
            'real': SimulationMode.REAL_NET
        }
        
        # Run experiments for each mode and latency combination
        latency_ious = []
        
        for mode_name in args.modes:
            mode = mode_map[mode_name]
            
            for latency_ms in args.latencies:
                # Update config with the latency
                config.network_delay = latency_ms / 1000.0  # Convert ms to seconds
                
                print(f'\n=== Running scenarios for {mode_name} mode, latency: {latency_ms}ms ===')
                
                ious = run_experiments_for_mode(
                    mode=mode,
                    config=config,
                    scenarios=SCENARIOS,
                    recording_file=recording_file,
                    verbose=True
                )
                
                latency_ious.append((mode_name, latency_ms, ious))
                
                # Print summary statistics
                if ious:
                    print(f'  Results: mean IOU = {np.mean(ious):.3f}, std = {np.std(ious):.3f}')
        
        # Generate visualization if not disabled  
        if not args.no_plot:
            plot_comparison_results(latency_ious)
        else:
            print('\nVisualization skipped (--no-plot option)')
            
        # Print comparison summary
        print("\n" + "="*70)
        print("PARKING PERFORMANCE COMPARISON")
        print("="*70)
        
        # Group results by mode for comparison
        mode_results = {}
        for mode_name, latency_ms, ious in latency_ious:
            if mode_name not in mode_results:
                mode_results[mode_name] = []
            mode_results[mode_name].extend(ious)
        
        # Print mode comparison
        for mode_name, all_ious in mode_results.items():
            if all_ious:
                print(f"\n{mode_name.upper()} mode:")
                print(f"  Mean IoU: {np.mean(all_ious):.3f} (±{np.std(all_ious):.3f})")
                print(f"  Min IoU: {np.min(all_ious):.3f}")
                print(f"  Max IoU: {np.max(all_ious):.3f}")
        
        # Compare virtual vs other modes if available
        if 'virtual' in mode_results and len(mode_results) > 1:
            virtual_mean = np.mean(mode_results['virtual'])
            print(f"\nPerformance comparison (baseline: Virtual Timeline):")
            
            for mode_name, all_ious in mode_results.items():
                if mode_name == 'virtual':
                    continue
                other_mean = np.mean(all_ious)
                diff = other_mean - virtual_mean
                print(f"  Virtual vs {mode_name}: {virtual_mean:.3f} vs {other_mean:.3f} ({diff:+.3f})")
        
    except KeyboardInterrupt:
        print('\nSimulation interrupted by user')
    except Exception as e:
        print(f'Error during simulation: {e}')
        raise
    finally:
        if recording_file:
            recording_file.close()


if __name__ == '__main__':
    main()