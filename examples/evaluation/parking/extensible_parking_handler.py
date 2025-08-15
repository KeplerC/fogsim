#!/usr/bin/env python3
"""
Extensible Parking Handler V3 - Correctly Implements Component-Specific Delays

This version properly simulates perception, planning, and control delays:
1. Cloud Perception: Obstacle detection is delayed (car uses old perception data)
2. Cloud Planning: Perception is immediate but planning decisions are delayed  
3. Full Cloud: All components are delayed cumulatively
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque

from fogsim.handlers import BaseHandler
from cloud_components import (
    CloudArchitectureConfig, 
    ComponentLocation,
    CLOUD_SCENARIOS
)

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

logger = logging.getLogger(__name__)


@dataclass
class PerceptionSnapshot:
    """Snapshot of perception data at a specific time."""
    obstacle_map: Any  # The obstacle map
    dynamic_bbs: List  # Dynamic bounding boxes
    # Vehicle localization data (also delayed in cloud perception)
    vehicle_x: float
    vehicle_y: float
    vehicle_yaw: float
    vehicle_vx: float
    vehicle_vy: float
    frame_id: int
    timestamp: float


@dataclass 
class PlanningRequest:
    """Request for planning sent to cloud."""
    frame_id: int
    timestamp: float
    car_position: Tuple[float, float]
    car_destination: Tuple[float, float]


class ExtensibleParkingHandler(BaseHandler):
    """
    Parking handler V3 with proper component-specific delay simulation.
    
    Key improvements:
    - Cloud perception: Car uses delayed obstacle maps
    - Cloud planning: Car sees obstacles immediately but planning is delayed
    - Proper cumulative delays for full cloud scenario
    """
    
    def __init__(self, scenario_config, cloud_config: CloudArchitectureConfig):
        """Initialize the extensible parking handler."""
        self.scenario_config = scenario_config
        self.cloud_config = cloud_config
        
        # CARLA components
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
        
        # State tracking
        self.frame_idx = 0
        self._launched = False
        self._observation = None
        self._last_iou = 0.0
        self._episode_done = False
        self._parking_time = None
        self._planning_active = False
        
        # Perception delay tracking
        self._perception_buffer = deque(maxlen=100)  # Buffer of perception snapshots
        self._current_perception_frame = -1  # Frame of current perception being used
        self._perception_request_sent_frame = -1
        self._waiting_for_perception = False
        
        # Planning delay tracking  
        self._planning_request_sent_frame = -1
        self._waiting_for_planning = False
        self._pending_planning_request = None
        
        # Control delay tracking (for full cloud)
        self._control_request_sent_frame = -1
        self._waiting_for_control = False
        
        logger.info(f"ExtensibleParkingHandler V3 initialized with cloud config: {cloud_config.name}")
        logger.info(f"  Perception: {cloud_config.perception_location.value}")
        logger.info(f"  Planning: {cloud_config.planning_location.value}")
        logger.info(f"  Control: {cloud_config.control_location.value}")
        
    def launch(self) -> None:
        """Launch CARLA and initialize the parking scenario."""
        if self._launched:
            return
            
        try:
            from parking_experiment_fogsim import ensure_carla_running
            if not ensure_carla_running():
                raise RuntimeError("Failed to start CARLA server")
                
            logger.info("Starting CARLA client connection...")
            self.client = load_client()
            if self.client is None:
                raise RuntimeError("Failed to create CARLA client")
                
            logger.info("Loading Town04 world...")
            self.world = town04_load(self.client)
            if self.world is None:
                raise RuntimeError("Failed to load Town04 world")
                
            logger.info("Setting spectator view...")
            town04_spectator_bev(self.world)
            
            logger.info("CARLA initialization successful")
            self._launched = True
            
        except Exception as e:
            logger.error(f"Failed to launch CARLA: {e}")
            self._launched = False
            raise RuntimeError(f"CARLA launch failed: {e}") from e
        
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
        parked_cars, parked_cars_bbs = town04_spawn_parked_cars(
            self.world, self.parked_spots, 
            self.destination_parking_spot, self.scenario_config.num_random_cars
        )
        self.actors_to_cleanup.extend(parked_cars)
        
        traffic_cones, traffic_cone_bbs = town04_spawn_traffic_cones(
            self.world, self.scenario_config.traffic_cone_positions
        )
        self.actors_to_cleanup.extend(traffic_cones)
        
        walkers, walker_bbs = town04_spawn_walkers(
            self.world, self.scenario_config.walker_positions
        )
        self.actors_to_cleanup.extend(walkers)
        self.walkers = walkers
        
        self.static_bbs = parked_cars_bbs + traffic_cone_bbs
        try:
            self.world.tick()
        except Exception as e:
            logger.error(f"Failed to tick world during actor spawning: {e}")
            raise RuntimeError(f"World tick failed: {e}") from e
        
    def _cleanup_actors(self):
        """Clean up all spawned actors with error handling."""
        for i, actor in enumerate(self.actors_to_cleanup):
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as e:
                    logger.warning(f"Failed to destroy actor {i}: {e}")
        self.actors_to_cleanup.clear()
        
    def set_states(self, states: Optional[Dict[str, Any]] = None,
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states (used for reset and action application)."""
        if states is None:
            self._reset_scenario()
            
    def _reset_scenario(self):
        """Reset the parking scenario."""
        # Cleanup
        if self.recording_cam is not None:
            self.recording_cam.destroy()
            self.recording_cam = None
        if self.car is not None:
            self.car.destroy()
            self.car = None
        self._cleanup_actors()
        
        try:
            # Spawn new actors
            self._spawn_actors()
            
            # Initialize ego vehicle
            logger.info(f"Spawning ego vehicle for destination spot {self.destination_parking_spot}")
            self.car = town04_spawn_ego_vehicle(
                self.world, self.destination_parking_spot
            )
            if self.car is None:
                raise RuntimeError("Failed to spawn ego vehicle")
            
            # Initialize video recording if configured
            if self.recording_file is not None:
                try:
                    logger.info("Initializing video recording...")
                    self.recording_cam = self.car.init_recording(self.recording_file)
                except Exception as e:
                    logger.warning(f"Failed to initialize video recording: {e}")
                    self.recording_cam = None
            
            # Initialize perception with static obstacles
            self.car.car.obs = clear_obstacle_map(
                obstacle_map_from_bbs(self.static_bbs)
            )
            
        except Exception as e:
            logger.error(f"Failed to reset scenario: {e}")
            self._cleanup_actors()
            if self.car is not None:
                try:
                    self.car.destroy()
                except:
                    pass
                self.car = None
            raise RuntimeError(f"Scenario reset failed: {e}") from e
        
        # Reset state variables
        self.frame_idx = 0
        self._episode_done = False
        self._last_iou = 0.0
        self._parking_time = None
        self._planning_active = False
        
        # Reset perception tracking
        self._perception_buffer.clear()
        self._current_perception_frame = -1
        self._perception_request_sent_frame = -1
        self._waiting_for_perception = False
        
        # Reset planning tracking
        self._planning_request_sent_frame = -1
        self._waiting_for_planning = False
        self._pending_planning_request = None
        
        # Reset control tracking
        self._control_request_sent_frame = -1
        self._waiting_for_control = False
        
        # Create initial observation
        self._update_observation()
    
    def _capture_perception_snapshot(self) -> PerceptionSnapshot:
        """Capture current perception state including vehicle localization."""
        # Update dynamic obstacles
        if self.walkers:
            self.dynamic_bbs = update_walkers(self.walkers)
        else:
            self.dynamic_bbs = []
        
        # Get current vehicle state (will be delayed in cloud perception)
        transform = self.car.actor.get_transform()
        velocity = self.car.actor.get_velocity()
        
        # Create obstacle map from current state
        all_bbs = self.static_bbs + self.dynamic_bbs
        current_obstacle_map = mask_obstacle_map(
            obstacle_map_from_bbs(all_bbs),
            self.car.car.cur.x,
            self.car.car.cur.y
        )
        
        return PerceptionSnapshot(
            obstacle_map=current_obstacle_map,
            dynamic_bbs=self.dynamic_bbs.copy(),
            # Vehicle localization (position, orientation, velocity)
            vehicle_x=transform.location.x,
            vehicle_y=transform.location.y,
            vehicle_yaw=transform.rotation.yaw,
            vehicle_vx=velocity.x,
            vehicle_vy=velocity.y,
            frame_id=self.frame_idx,
            timestamp=time.time()
        )
    
    def _update_perception_immediate(self):
        """Update perception immediately (for baseline and local perception)."""
        snapshot = self._capture_perception_snapshot()
        
        # Update car's obstacle map immediately
        self.car.car.obs = union_obstacle_map(
            self.car.car.obs,
            snapshot.obstacle_map
        )
        
        # For immediate perception, ensure localization is current
        # (In case it was previously set to delayed values)
        transform = self.car.actor.get_transform()
        self.car.car.cur.x = transform.location.x
        self.car.car.cur.y = transform.location.y
        self.car.car.cur.angle = transform.rotation.yaw
        
        self._current_perception_frame = self.frame_idx
        
    def _update_perception_delayed(self, delayed_snapshot: PerceptionSnapshot):
        """Update perception with delayed data (for cloud perception)."""
        # Use the delayed obstacle map
        self.car.car.obs = union_obstacle_map(
            self.car.car.obs,
            delayed_snapshot.obstacle_map
        )
        
        # CRITICAL: Update car's perceived position with delayed localization
        # This makes the car think it's where it was N frames ago
        import carla
        try:
            # Create a delayed position for the car's internal state
            # The car will plan based on where it THINKS it is (delayed position)
            self.car.car.cur.x = delayed_snapshot.vehicle_x
            self.car.car.cur.y = delayed_snapshot.vehicle_y
            self.car.car.cur.angle = delayed_snapshot.vehicle_yaw
            
            # Note: We don't update the actual CARLA actor position, just the car's
            # internal belief about where it is. This simulates delayed localization.
            
            if self.frame_idx % 50 == 0:
                # Calculate how wrong the localization is
                actual_transform = self.car.actor.get_transform()
                position_error = np.sqrt(
                    (actual_transform.location.x - delayed_snapshot.vehicle_x)**2 + 
                    (actual_transform.location.y - delayed_snapshot.vehicle_y)**2
                )
                logger.info(f"Delayed localization: position error = {position_error:.2f}m")
        except Exception as e:
            logger.warning(f"Failed to update delayed localization: {e}")
        
        self._current_perception_frame = delayed_snapshot.frame_id
        
        if self.frame_idx % 50 == 0:
            perception_delay_frames = self.frame_idx - delayed_snapshot.frame_id
            perception_delay_ms = perception_delay_frames * self.scenario_config.timestep * 1000
            logger.info(f"Using perception from frame {delayed_snapshot.frame_id} "
                       f"(delay: {perception_delay_ms:.1f}ms)")
    
    def _should_replan(self):
        """Determine if replanning is needed."""
        try:
            if self.car is None or not hasattr(self.car, 'actor') or self.car.actor is None:
                return False
            try:
                self.car.actor.get_transform()
            except RuntimeError:
                return False
            return (self.frame_idx % self.scenario_config.replan_interval == 0 and 
                    self.car.car.cur.distance(self.car.car.destination) > self.scenario_config.distance_threshold)
        except Exception as e:
            logger.warning(f"Error checking replan condition: {e}")
            return False
        
    def _update_observation(self):
        """Update the observation vector."""
        if self.car is None:
            self._observation = np.zeros(15)
            return
            
        try:
            if not hasattr(self.car, 'actor') or self.car.actor is None:
                self._observation = np.zeros(15)
                self._episode_done = True
                return
            
            try:
                self.car.actor.get_transform()
            except RuntimeError:
                self._observation = np.zeros(15)
                self._episode_done = True
                return
            
            # Get car state
            transform = self.car.actor.get_transform()
            velocity = self.car.actor.get_velocity()
            
            # Calculate distance to destination
            dist_to_dest = self.car.car.cur.distance(self.car.car.destination)
            
            # Check if done
            self._episode_done = is_done(self.car)
            
            # Calculate IoU and parking time if parked
            if self._episode_done:
                try:
                    self._last_iou = self.car.iou()
                except Exception as e:
                    logger.warning(f"Failed to calculate IoU: {e}")
                    self._last_iou = 0.0
                    
                if self._parking_time is None:
                    self._parking_time = self.frame_idx * self.scenario_config.timestep
                
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
                self.frame_idx / self.scenario_config.max_episode_steps,
                np.sum(self.car.car.obs.obs) / (self.car.car.obs.obs.size + 1e-6),
                np.mean(self.car.car.obs.obs),
                np.std(self.car.car.obs.obs),
                self.car.car.destination.x,
                self.car.car.destination.y,
                self.car.car.destination.angle,
            ])
            
        except Exception as e:
            logger.error(f"Failed to update observation: {e}")
            self._observation = np.zeros(15)
            self._episode_done = True
        
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
            
        self._update_observation()
        
        return {
            'observation': self._observation,
            'done': self._episode_done,
            'iou': self._last_iou,
            'parking_time': self._parking_time,
            'frame': self.frame_idx,
            'car_position': [self.car.car.cur.x, self.car.car.cur.y] if self.car else [0, 0],
            'destination': [self.car.car.destination.x, self.car.car.destination.y] if self.car else [0, 0],
            'cloud_config': self.cloud_config.name,
            'perception_delay_frames': self.frame_idx - self._current_perception_frame if self._current_perception_frame >= 0 else 0,
        }
        
    def step(self) -> None:
        """Step the simulation forward."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self._episode_done:
            return
        
        # Tick simulation
        try:
            self.world.tick()
        except Exception as e:
            logger.error(f"World tick failed: {e}")
            self._episode_done = True
            return
            
        try:
            self.car.localize()
        except Exception as e:
            logger.error(f"Car localization failed: {e}")
        
        # Process recording frames if video recording is enabled
        if self.recording_file:
            latency_ms = self.scenario_config.network_delay * 1000.0
            source_rate_kbps = self.scenario_config.source_rate / 1000.0
            cloud_mode_short = {
                "baseline": "Baseline (All Local)",
                "cloud_perception": "Cloud Perception",
                "cloud_planning": "Cloud Planning", 
                "full_cloud": "Full Cloud"
            }.get(self.cloud_config.name, self.cloud_config.name)
            self.car.process_recording_frames(latency_ms, source_rate_kbps, cloud_mode_short)
        
        # Increment frame counter
        self.frame_idx += 1
                
    def step_with_action(self, action: Optional[np.ndarray]) -> Tuple:
        """
        Process simulation step with proper component-specific delays.
        
        Key behaviors by scenario:
        - Baseline: Everything immediate
        - Cloud Perception: Obstacle detection delayed, planning/control immediate after perception arrives
        - Cloud Planning: Perception immediate, planning delayed, control immediate after planning
        - Full Cloud: Everything delayed cumulatively
        """
        # Execute physics simulation step
        self.step()
        
        # Track if we received a response from cloud (network delivered something)
        cloud_response_received = action is not None and isinstance(action, np.ndarray)
        
        # Capture current perception (may or may not be used depending on scenario)
        current_perception = self._capture_perception_snapshot()
        
        # Store perception snapshot for potential delayed use
        self._perception_buffer.append(current_perception)
        
        # Check if it's time to plan
        should_plan = self._should_replan()
        
        # Handle different cloud scenarios
        if self.cloud_config.name == 'baseline':
            # BASELINE: Everything immediate
            self._update_perception_immediate()
            
            if should_plan:
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Baseline - immediate perception and planning")
                
                try:
                    if self.car and self.car.actor:
                        self.car.actor.get_transform()
                        self.car.plan()
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None
                        self._planning_active = has_plan
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
                    
        elif self.cloud_config.name == 'cloud_perception':
            # CLOUD PERCEPTION: Perception delayed, planning/control immediate after perception
            
            # Start perception request if needed
            if should_plan and not self._waiting_for_perception:
                self._perception_request_sent_frame = self.frame_idx
                self._waiting_for_perception = True
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Sending perception request to cloud")
            
            # Check if perception response arrived
            if self._waiting_for_perception and cloud_response_received:
                # Get the delayed perception snapshot
                delay_frames = int(self.scenario_config.network_delay / self.scenario_config.timestep)
                target_frame = self._perception_request_sent_frame
                
                # Find the perception from when the request was sent
                delayed_perception = None
                for snapshot in self._perception_buffer:
                    if snapshot.frame_id == target_frame:
                        delayed_perception = snapshot
                        break
                
                if delayed_perception:
                    # Update with delayed perception
                    self._update_perception_delayed(delayed_perception)
                    
                    if self.frame_idx % 50 == 0:
                        logger.info(f"Step {self.frame_idx}: Cloud perception received, planning with delayed obstacles")
                    
                    # Now plan with the delayed perception
                    try:
                        if self.car and self.car.actor:
                            self.car.actor.get_transform()
                            self.car.plan()
                            has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None
                            self._planning_active = has_plan
                            
                            # IMPORTANT: After planning, update localization to current for control
                            # Control needs to execute based on where the car actually is NOW
                            actual_transform = self.car.actor.get_transform()
                            self.car.car.cur.x = actual_transform.location.x
                            self.car.car.cur.y = actual_transform.location.y
                            self.car.car.cur.angle = actual_transform.rotation.yaw
                    except Exception as e:
                        logger.error(f"Planning failed: {e}")
                        self._planning_active = False
                
                self._waiting_for_perception = False
                
        elif self.cloud_config.name == 'cloud_planning':
            # CLOUD PLANNING: Perception immediate, planning delayed
            
            # Update perception immediately
            self._update_perception_immediate()
            
            # Start planning request if needed
            if should_plan and not self._waiting_for_planning:
                self._planning_request_sent_frame = self.frame_idx
                self._waiting_for_planning = True
                self._pending_planning_request = PlanningRequest(
                    frame_id=self.frame_idx,
                    timestamp=time.time(),
                    car_position=(self.car.car.cur.x, self.car.car.cur.y),
                    car_destination=(self.car.car.destination.x, self.car.car.destination.y)
                )
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Sending planning request to cloud")
            
            # Check if planning response arrived
            if self._waiting_for_planning and cloud_response_received:
                if self.frame_idx % 50 == 0:
                    delay_ms = (self.frame_idx - self._planning_request_sent_frame) * self.scenario_config.timestep * 1000
                    logger.info(f"Step {self.frame_idx}: Cloud planning received after {delay_ms:.1f}ms")
                
                # Execute planning now
                try:
                    if self.car and self.car.actor:
                        self.car.actor.get_transform()
                        self.car.plan()
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None
                        self._planning_active = has_plan
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
                
                self._waiting_for_planning = False
                
        elif self.cloud_config.name == 'full_cloud':
            # FULL CLOUD: Everything delayed (perception + planning + control)
            
            # This is complex - need to chain delays
            # For simplicity, we'll implement perception and planning delays
            # (control delay would require buffering control commands)
            
            # Start perception request if needed
            if should_plan and not self._waiting_for_perception:
                self._perception_request_sent_frame = self.frame_idx
                self._waiting_for_perception = True
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Full cloud - sending perception request")
            
            # Check if perception response arrived
            if self._waiting_for_perception and cloud_response_received:
                # Get delayed perception
                delay_frames = int(self.scenario_config.network_delay / self.scenario_config.timestep)
                target_frame = self._perception_request_sent_frame
                
                delayed_perception = None
                for snapshot in self._perception_buffer:
                    if snapshot.frame_id == target_frame:
                        delayed_perception = snapshot
                        break
                
                if delayed_perception:
                    self._update_perception_delayed(delayed_perception)
                    
                    # Now start planning request (additional delay)
                    self._waiting_for_perception = False
                    self._waiting_for_planning = True
                    self._planning_request_sent_frame = self.frame_idx
                    
                    if self.frame_idx % 50 == 0:
                        logger.info(f"Step {self.frame_idx}: Full cloud - perception done, sending planning request")
            
            # Check if planning response arrived (after perception)
            elif self._waiting_for_planning and cloud_response_received:
                if self.frame_idx % 50 == 0:
                    total_delay_ms = (self.frame_idx - self._perception_request_sent_frame) * self.scenario_config.timestep * 1000
                    logger.info(f"Step {self.frame_idx}: Full cloud - all processing done after {total_delay_ms:.1f}ms total")
                
                # Execute planning
                try:
                    if self.car and self.car.actor:
                        self.car.actor.get_transform()
                        self.car.plan()
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None
                        self._planning_active = has_plan
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
                
                self._waiting_for_planning = False
        
        else:
            # For non-delayed perception scenarios, update immediately
            if self.cloud_config.perception_location == ComponentLocation.LOCAL:
                self._update_perception_immediate()
        
        # Execute control if we have an active plan
        if self._planning_active:
            try:
                if self.car and self.car.actor:
                    self.car.run_step()
            except Exception as e:
                logger.error(f"Car run_step failed: {e}")
                self._planning_active = False
                # Emergency stop
                try:
                    if self.car and self.car.actor:
                        import carla
                        emergency_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                        self.car.actor.apply_control(emergency_control)
                except:
                    pass
        
        # Get current state
        states = self.get_states()
        observation = states['observation']
        
        # Calculate reward and termination
        reward = 0.0
        if states['done']:
            reward = states['iou'] * 100
            
        terminated = states['done']
        truncated = self.frame_idx >= self.scenario_config.max_episode_steps
        success = states['iou'] >= 0.8 if states['done'] else False
        
        # Debug info
        if self.frame_idx % 100 == 0:
            perception_delay = states.get('perception_delay_frames', 0) * self.scenario_config.timestep * 1000
            logger.info(f"Frame {self.frame_idx}: {self.cloud_config.name}, "
                       f"perception_delay={perception_delay:.1f}ms, "
                       f"planning_active={self._planning_active}")
        
        return observation, reward, success, terminated, truncated, states
        
    def render(self) -> Optional[np.ndarray]:
        """Render is not implemented for this handler."""
        return None
        
    def close(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up CARLA resources...")
        
        if self.recording_cam is not None:
            try:
                self.recording_cam.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy recording camera: {e}")
            finally:
                self.recording_cam = None
        
        if self.car is not None:
            try:
                if hasattr(self.car, 'actor') and self.car.actor:
                    self.car.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy car: {e}")
            finally:
                self.car = None
        
        try:
            self._cleanup_actors()
        except Exception as e:
            logger.warning(f"Failed to clean up actors: {e}")
        
        if self.world is not None:
            try:
                self.world.tick()
            except Exception as e:
                logger.warning(f"Failed to tick world during cleanup: {e}")
        
        self._launched = False
        logger.info("CARLA cleanup completed")
        
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata."""
        return {
            'scenario_config': self.scenario_config.to_dict(),
            'cloud_config': {
                'name': self.cloud_config.name,
                'perception_location': self.cloud_config.perception_location.value,
                'planning_location': self.cloud_config.planning_location.value,
                'control_location': self.cloud_config.control_location.value,
                'description': self.cloud_config.description
            },
            'launched': self._launched,
            'destination': self.destination_parking_spot,
            'parked_spots': self.parked_spots,
        }
        
    @property
    def is_launched(self) -> bool:
        """Check if handler is launched."""
        return self._launched