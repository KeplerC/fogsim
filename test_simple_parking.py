#!/usr/bin/env python3
"""
Simple test to verify the car actually moves in parking scenarios.
"""

import numpy as np
import time
from fogsim import FogSim, SimulationMode, NetworkConfig
from examples.evaluation.parking.cloud_components import CLOUD_SCENARIOS
from examples.evaluation.parking.extensible_parking_handler import ExtensibleParkingHandler
from examples.evaluation.parking.parking_experiment_extensible import ExtensibleScenarioConfig

def test_car_movement():
    """Test that the car actually moves."""
    
    print("\n" + "="*60)
    print("TESTING CAR MOVEMENT")
    print("="*60)
    
    # Simple scenario config
    scenario_config = ExtensibleScenarioConfig(
        name="test_movement",
        network_delay=0.01,  # 10ms
        packet_loss_rate=0.0,
        source_rate=1e6,
        timestep=0.05,
        num_random_cars=2,
        replan_interval=1,  # Plan every frame for testing
        distance_threshold=5.0,
        max_episode_steps=200,
    )
    
    # Test baseline first (should work)
    cloud_config = CLOUD_SCENARIOS['baseline']
    
    print(f"\nTesting {cloud_config.name}...")
    
    # Create handler
    handler = ExtensibleParkingHandler(scenario_config, cloud_config)
    handler.set_scenario(destination=20, parked_spots=[19])
    
    # Configure network
    network_config = NetworkConfig()
    network_config.topology.link_delay = scenario_config.network_delay
    network_config.source_rate = scenario_config.source_rate
    
    # Create FogSim
    fogsim = FogSim(
        handler,
        mode=SimulationMode.VIRTUAL,
        timestep=scenario_config.timestep,
        network_config=network_config
    )
    
    # Reset
    obs, info = fogsim.reset()
    
    # Track car position
    positions = []
    
    # Run for a few steps
    for step in range(100):
        # For baseline, no action needed
        # For cloud, we'd send the observation
        action = None
        
        # Step
        obs, reward, success, terminated, truncated, info = fogsim.step(action)
        
        # Track position
        if 'car_position' in info:
            positions.append(info['car_position'])
        
        # Check every 20 steps
        if step % 20 == 0:
            if positions:
                current_pos = positions[-1]
                print(f"  Step {step}: Car at ({current_pos[0]:.1f}, {current_pos[1]:.1f})")
                
                # Check if car has moved
                if len(positions) > 1:
                    prev_pos = positions[0]
                    distance_moved = np.sqrt((current_pos[0] - prev_pos[0])**2 + 
                                           (current_pos[1] - prev_pos[1])**2)
                    if distance_moved > 0.1:
                        print(f"    ✓ Car has moved {distance_moved:.1f} meters")
                    else:
                        print(f"    ✗ Car hasn't moved!")
        
        if terminated or truncated:
            break
    
    # Clean up
    fogsim.close()
    
    # Check if car moved
    if len(positions) > 1:
        start_pos = positions[0]
        end_pos = positions[-1]
        total_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + 
                                (end_pos[1] - start_pos[1])**2)
        
        print(f"\nResults:")
        print(f"  Start position: ({start_pos[0]:.1f}, {start_pos[1]:.1f})")
        print(f"  End position: ({end_pos[0]:.1f}, {end_pos[1]:.1f})")
        print(f"  Total distance moved: {total_distance:.1f} meters")
        
        if total_distance > 1.0:
            print("  ✓ Car movement verified!")
        else:
            print("  ✗ Car didn't move enough!")
    else:
        print("  ✗ No position data collected!")

if __name__ == '__main__':
    test_car_movement()