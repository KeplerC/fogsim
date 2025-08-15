#!/usr/bin/env python3
"""
Cloud Components for Extensible Parking Simulation

This module defines the base components (perception, planning, control) that can run
locally or on the cloud with network delays through FogSim.

Architecture supports three scenarios:
1. Cloud Perception: Vehicle → Cloud Perception (delayed) → Local Planning → Local Control
2. Cloud Planning: Local Perception → Cloud Planning (delayed) → Local Control
3. Full Cloud: Vehicle → Cloud Perception → Cloud Planning → Cloud Control (delayed)
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Import parking-specific utilities
from experiment_utils import (
    update_walkers,
    obstacle_map_from_bbs,
    clear_obstacle_map,
    union_obstacle_map,
    mask_obstacle_map,
)


class ComponentLocation(Enum):
    """Where a component runs."""
    LOCAL = "local"      # On vehicle (no delay)
    CLOUD = "cloud"      # On cloud (with network delay)


@dataclass
class PerceptionData:
    """Perception output data."""
    obstacle_map: np.ndarray
    vehicle_position: Tuple[float, float, float]  # x, y, yaw
    vehicle_velocity: Tuple[float, float]  # vx, vy
    timestamp: float
    frame_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'obstacle_map': self.obstacle_map.tolist(),
            'vehicle_position': self.vehicle_position,
            'vehicle_velocity': self.vehicle_velocity,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerceptionData':
        return cls(
            obstacle_map=np.array(data['obstacle_map']),
            vehicle_position=tuple(data['vehicle_position']),
            vehicle_velocity=tuple(data['vehicle_velocity']),
            timestamp=data['timestamp'],
            frame_id=data['frame_id']
        )


@dataclass 
class PlanningData:
    """Planning output data."""
    trajectory: List[Tuple[float, float, float]]  # List of (x, y, yaw) waypoints
    target_speed: float
    steering_angle: float
    has_plan: bool
    timestamp: float
    frame_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trajectory': self.trajectory,
            'target_speed': self.target_speed,
            'steering_angle': self.steering_angle,
            'has_plan': self.has_plan,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PlanningData':
        return cls(
            trajectory=data['trajectory'],
            target_speed=data['target_speed'],
            steering_angle=data['steering_angle'],
            has_plan=data['has_plan'],
            timestamp=data['timestamp'],
            frame_id=data['frame_id']
        )


@dataclass
class ControlData:
    """Control output data."""
    throttle: float
    brake: float
    steer: float
    timestamp: float
    frame_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'throttle': self.throttle,
            'brake': self.brake,
            'steer': self.steer,
            'timestamp': self.timestamp,
            'frame_id': self.frame_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ControlData':
        return cls(
            throttle=data['throttle'],
            brake=data['brake'],
            steer=data['steer'],
            timestamp=data['timestamp'],
            frame_id=data['frame_id']
        )


class BasePerceptionComponent(ABC):
    """Base class for perception components."""
    
    def __init__(self, location: ComponentLocation):
        self.location = location
        self.frame_id = 0
        
    @abstractmethod
    def process(self, car, static_bbs: List, dynamic_bbs: List) -> PerceptionData:
        """Process raw sensor data into perception output."""
        pass


class BaseplanningComponent(ABC):
    """Base class for planning components."""
    
    def __init__(self, location: ComponentLocation):
        self.location = location
        self.frame_id = 0
        
    @abstractmethod  
    def process(self, perception_data: PerceptionData, car) -> PlanningData:
        """Process perception data into planning output."""
        pass


class BaseControlComponent(ABC):
    """Base class for control components."""
    
    def __init__(self, location: ComponentLocation):
        self.location = location
        self.frame_id = 0
        
    @abstractmethod
    def process(self, planning_data: PlanningData, car) -> ControlData:
        """Process planning data into control commands."""
        pass


class LocalPerceptionComponent(BasePerceptionComponent):
    """Local perception component (runs on vehicle)."""
    
    def process(self, car, static_bbs: List, dynamic_bbs: List) -> PerceptionData:
        """Process perception locally on vehicle."""
        # Update perception with current obstacles
        all_bbs = static_bbs + dynamic_bbs
        
        # Create obstacle map from bounding boxes
        obstacle_map_raw = obstacle_map_from_bbs(all_bbs)
        masked_map = mask_obstacle_map(
            obstacle_map_raw,
            car.car.cur.x,
            car.car.cur.y
        )
        
        # Union with existing obstacle map
        car.car.obs = union_obstacle_map(car.car.obs, masked_map)
        
        # Get vehicle state
        transform = car.actor.get_transform()
        velocity = car.actor.get_velocity()
        
        self.frame_id += 1
        return PerceptionData(
            obstacle_map=car.car.obs.obs.copy(),
            vehicle_position=(transform.location.x, transform.location.y, transform.rotation.yaw),
            vehicle_velocity=(velocity.x, velocity.y),
            timestamp=time.time(),
            frame_id=self.frame_id
        )


class CloudPerceptionComponent(BasePerceptionComponent):
    """Cloud perception component (runs on cloud with network delay)."""
    
    def process(self, car, static_bbs: List, dynamic_bbs: List) -> PerceptionData:
        """Process perception on cloud (same logic but will have network delay)."""
        # Same processing as local, but this will be called after network delay
        all_bbs = static_bbs + dynamic_bbs
        
        # Create obstacle map from bounding boxes
        obstacle_map_raw = obstacle_map_from_bbs(all_bbs)
        masked_map = mask_obstacle_map(
            obstacle_map_raw,
            car.car.cur.x,
            car.car.cur.y
        )
        
        # Union with existing obstacle map
        car.car.obs = union_obstacle_map(car.car.obs, masked_map)
        
        # Get vehicle state
        transform = car.actor.get_transform()
        velocity = car.actor.get_velocity()
        
        self.frame_id += 1
        return PerceptionData(
            obstacle_map=car.car.obs.obs.copy(),
            vehicle_position=(transform.location.x, transform.location.y, transform.rotation.yaw),
            vehicle_velocity=(velocity.x, velocity.y),
            timestamp=time.time(),
            frame_id=self.frame_id
        )


class LocalPlanningComponent(BaseplanningComponent):
    """Local planning component (runs on vehicle)."""
    
    def process(self, perception_data: PerceptionData, car) -> PlanningData:
        """Process planning locally on vehicle."""
        # Update car's perception with the provided data
        # Make sure the obstacle map has the right structure
        if perception_data.obstacle_map.size > 0:
            car.car.obs.obs = perception_data.obstacle_map
        
        # Plan every time like the original - let the car itself decide when to replan
        should_plan = (hasattr(car.car, 'cur') and hasattr(car.car, 'destination'))
        
        has_plan = False
        trajectory = []
        target_speed = 0.0
        steering_angle = 0.0
        
        if should_plan:
            try:
                # Debug: Check current position and destination
                if self.frame_id % 100 == 0:  # Log every 100 frames
                    print(f"Planning: pos=({car.car.cur.x:.1f},{car.car.cur.y:.1f}) dest=({car.car.destination.x:.1f},{car.car.destination.y:.1f}) dist={car.car.cur.distance(car.car.destination):.1f}")
                
                # Run planner
                car.plan()
                
                # Extract planning results
                has_plan = hasattr(car.car, 'path') and car.car.path is not None and len(car.car.path) > 0
                
                if self.frame_id % 100 == 0:  # Log planning results
                    print(f"Planning result: has_plan={has_plan}, path_length={len(car.car.path) if has_plan else 0}")
                
                if has_plan:
                    # Extract waypoints from path
                    trajectory = [(wp.x, wp.y, wp.angle) for wp in car.car.path[:10]]  # Next 10 waypoints
                    
                    # Get control parameters (use defaults if not available)
                    target_speed = getattr(car.car, 'speed', 2.0)  # Default speed
                    steering_angle = getattr(car.car, 'steering', 0.0)  # Default steering
                    
            except Exception as e:
                print(f"Planning failed: {e}")
                has_plan = False
        
        self.frame_id += 1
        return PlanningData(
            trajectory=trajectory,
            target_speed=target_speed,
            steering_angle=steering_angle,
            has_plan=has_plan,
            timestamp=time.time(),
            frame_id=self.frame_id
        )


class CloudPlanningComponent(BaseplanningComponent):
    """Cloud planning component (runs on cloud with network delay)."""
    
    def process(self, perception_data: PerceptionData, car) -> PlanningData:
        """Process planning on cloud (same logic but will have network delay)."""
        # Same processing as local planning
        if perception_data.obstacle_map.size > 0:
            car.car.obs.obs = perception_data.obstacle_map
        
        # Plan every time like the original - let the car itself decide when to replan
        should_plan = (hasattr(car.car, 'cur') and hasattr(car.car, 'destination'))
        
        has_plan = False
        trajectory = []
        target_speed = 0.0
        steering_angle = 0.0
        
        if should_plan:
            try:
                # Debug: Check current position and destination
                if self.frame_id % 100 == 0:  # Log every 100 frames
                    print(f"Planning: pos=({car.car.cur.x:.1f},{car.car.cur.y:.1f}) dest=({car.car.destination.x:.1f},{car.car.destination.y:.1f}) dist={car.car.cur.distance(car.car.destination):.1f}")
                
                # Run planner
                car.plan()
                
                # Extract planning results
                has_plan = hasattr(car.car, 'path') and car.car.path is not None and len(car.car.path) > 0
                
                if self.frame_id % 100 == 0:  # Log planning results
                    print(f"Planning result: has_plan={has_plan}, path_length={len(car.car.path) if has_plan else 0}")
                
                if has_plan:
                    # Extract waypoints from path
                    trajectory = [(wp.x, wp.y, wp.angle) for wp in car.car.path[:10]]  # Next 10 waypoints
                    
                    # Get control parameters (use defaults if not available)
                    target_speed = getattr(car.car, 'speed', 2.0)  # Default speed
                    steering_angle = getattr(car.car, 'steering', 0.0)  # Default steering
                    
            except Exception as e:
                print(f"Cloud planning failed: {e}")
                has_plan = False
        
        self.frame_id += 1
        return PlanningData(
            trajectory=trajectory,
            target_speed=target_speed,
            steering_angle=steering_angle,
            has_plan=has_plan,
            timestamp=time.time(),
            frame_id=self.frame_id
        )


class LocalControlComponent(BaseControlComponent):
    """Local control component (runs on vehicle)."""
    
    def process(self, planning_data: PlanningData, car) -> ControlData:
        """Process control locally on vehicle."""
        # Apply the planned control
        if planning_data.has_plan:
            # Use the planning results to generate control commands
            car.run_step()  # Execute the planned trajectory
            
            # Get the applied control (extract from car's last control)
            control = car.actor.get_control() if hasattr(car.actor, 'get_control') else None
            if control:
                throttle = control.throttle
                brake = control.brake
                steer = control.steer
            else:
                # Fallback: use planning data
                throttle = min(planning_data.target_speed / 10.0, 1.0)  # Simple speed-to-throttle
                brake = 0.0
                steer = planning_data.steering_angle
        else:
            # No plan - stop
            throttle = 0.0
            brake = 0.5
            steer = 0.0
            
            # Apply control directly
            try:
                import carla
                car.actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=steer))
            except:
                pass
        
        self.frame_id += 1
        return ControlData(
            throttle=throttle,
            brake=brake,
            steer=steer,
            timestamp=time.time(),
            frame_id=self.frame_id
        )


class CloudControlComponent(BaseControlComponent):
    """Cloud control component (runs on cloud with network delay)."""
    
    def process(self, planning_data: PlanningData, car) -> ControlData:
        """Process control on cloud (commands will be delayed)."""
        # Generate control commands (same logic as local)
        if planning_data.has_plan:
            throttle = min(planning_data.target_speed / 10.0, 1.0)
            brake = 0.0
            steer = planning_data.steering_angle
        else:
            throttle = 0.0
            brake = 0.5
            steer = 0.0
        
        self.frame_id += 1
        return ControlData(
            throttle=throttle,
            brake=brake,
            steer=steer,
            timestamp=time.time(),
            frame_id=self.frame_id
        )


@dataclass
class CloudArchitectureConfig:
    """Configuration for cloud architecture scenarios."""
    name: str
    perception_location: ComponentLocation
    planning_location: ComponentLocation
    control_location: ComponentLocation
    description: str
    
    def get_components(self) -> Tuple[BasePerceptionComponent, BaseplanningComponent, BaseControlComponent]:
        """Create component instances based on configuration."""
        # Perception component
        if self.perception_location == ComponentLocation.LOCAL:
            perception = LocalPerceptionComponent(self.perception_location)
        else:
            perception = CloudPerceptionComponent(self.perception_location)
        
        # Planning component
        if self.planning_location == ComponentLocation.LOCAL:
            planning = LocalPlanningComponent(self.planning_location)
        else:
            planning = CloudPlanningComponent(self.planning_location)
        
        # Control component
        if self.control_location == ComponentLocation.LOCAL:
            control = LocalControlComponent(self.control_location)
        else:
            control = CloudControlComponent(self.control_location)
        
        return perception, planning, control


# Predefined cloud architecture scenarios
CLOUD_SCENARIOS = {
    "baseline": CloudArchitectureConfig(
        name="baseline",
        perception_location=ComponentLocation.LOCAL,
        planning_location=ComponentLocation.LOCAL,
        control_location=ComponentLocation.LOCAL,
        description="All processing on vehicle (baseline, no cloud)"
    ),
    
    "cloud_perception": CloudArchitectureConfig(
        name="cloud_perception",
        perception_location=ComponentLocation.CLOUD,
        planning_location=ComponentLocation.LOCAL,
        control_location=ComponentLocation.LOCAL,
        description="Perception on cloud (delayed), planning and control local"
    ),
    
    "cloud_planning": CloudArchitectureConfig(
        name="cloud_planning",
        perception_location=ComponentLocation.CLOUD,
        planning_location=ComponentLocation.CLOUD,
        control_location=ComponentLocation.LOCAL,
        description="Perception and planning on cloud (delayed), control local"
    ),
    
    "full_cloud": CloudArchitectureConfig(
        name="full_cloud",
        perception_location=ComponentLocation.CLOUD,
        planning_location=ComponentLocation.CLOUD,
        control_location=ComponentLocation.CLOUD,
        description="All processing on cloud (all delayed)"
    ),
}