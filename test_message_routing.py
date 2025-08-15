#!/usr/bin/env python3
"""
Simple unit test to verify message routing logic without CARLA.
"""

import sys
import os
sys.path.append('/home/kych/cloudsim/examples/evaluation/parking')

import numpy as np
import json
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import cloud components
from cloud_components import (
    CloudArchitectureConfig,
    ComponentLocation,
    PerceptionData,
    PlanningData,
    ControlData,
    CLOUD_SCENARIOS
)


def test_cloud_message_routing():
    """Test that different cloud scenarios route messages correctly."""
    
    print("\n" + "="*60)
    print("MESSAGE ROUTING TEST")
    print("="*60)
    
    # Test each cloud scenario
    for scenario_name, cloud_config in CLOUD_SCENARIOS.items():
        print(f"\n{scenario_name}:")
        print(f"  Description: {cloud_config.description}")
        print(f"  Perception: {cloud_config.perception_location.value}")
        print(f"  Planning: {cloud_config.planning_location.value}")
        print(f"  Control: {cloud_config.control_location.value}")
        
        # Verify component locations
        if scenario_name == 'baseline':
            assert cloud_config.perception_location == ComponentLocation.LOCAL
            assert cloud_config.planning_location == ComponentLocation.LOCAL
            assert cloud_config.control_location == ComponentLocation.LOCAL
            print("  ✓ All components local (no cloud delay)")
            
        elif scenario_name == 'cloud_perception':
            assert cloud_config.perception_location == ComponentLocation.CLOUD
            assert cloud_config.planning_location == ComponentLocation.LOCAL
            assert cloud_config.control_location == ComponentLocation.LOCAL
            print("  ✓ Perception on cloud (delayed), planning/control local")
            
        elif scenario_name == 'cloud_planning':
            assert cloud_config.perception_location == ComponentLocation.CLOUD
            assert cloud_config.planning_location == ComponentLocation.CLOUD
            assert cloud_config.control_location == ComponentLocation.LOCAL
            print("  ✓ Perception/planning on cloud (delayed), control local")
            
        elif scenario_name == 'full_cloud':
            assert cloud_config.perception_location == ComponentLocation.CLOUD
            assert cloud_config.planning_location == ComponentLocation.CLOUD
            assert cloud_config.control_location == ComponentLocation.CLOUD
            print("  ✓ All components on cloud (fully delayed)")
    
    print("\n" + "="*60)
    print("MESSAGE SERIALIZATION TEST")
    print("="*60)
    
    # Test message serialization
    perception_data = PerceptionData(
        obstacle_map=np.zeros((10, 10)),
        vehicle_position=(100.0, 200.0, 45.0),
        vehicle_velocity=(5.0, 0.0),
        timestamp=123.456,
        frame_id=42
    )
    
    # Convert to dict and back
    perception_dict = perception_data.to_dict()
    perception_restored = PerceptionData.from_dict(perception_dict)
    
    assert perception_restored.vehicle_position == perception_data.vehicle_position
    assert perception_restored.frame_id == perception_data.frame_id
    print("✓ PerceptionData serialization works")
    
    planning_data = PlanningData(
        trajectory=[(100, 200, 45), (110, 200, 45)],
        target_speed=5.0,
        steering_angle=0.1,
        has_plan=True,
        timestamp=123.456,
        frame_id=42
    )
    
    planning_dict = planning_data.to_dict()
    planning_restored = PlanningData.from_dict(planning_dict)
    
    assert planning_restored.trajectory == planning_data.trajectory
    assert planning_restored.has_plan == planning_data.has_plan
    print("✓ PlanningData serialization works")
    
    control_data = ControlData(
        throttle=0.5,
        brake=0.0,
        steer=0.1,
        timestamp=123.456,
        frame_id=42
    )
    
    control_dict = control_data.to_dict()
    control_restored = ControlData.from_dict(control_dict)
    
    assert control_restored.throttle == control_data.throttle
    assert control_restored.frame_id == control_data.frame_id
    print("✓ ControlData serialization works")
    
    print("\n" + "="*60)
    print("LATENCY EXPECTATIONS")
    print("="*60)
    
    print("\nExpected latency behavior for 50ms network delay:")
    print("  Baseline:          0ms (all local)")
    print("  Cloud Perception:  ~50ms for perception, then local")
    print("  Cloud Planning:    ~50ms for perception + ~50ms for planning")
    print("  Full Cloud:        ~50ms for each component (cumulative)")
    
    print("\n✓ All message routing tests passed!")
    

if __name__ == '__main__':
    test_cloud_message_routing()