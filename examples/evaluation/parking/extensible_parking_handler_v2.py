#!/usr/bin/env python3
"""
Extensible Parking Handler V2 - Properly Simulates Cloud Delays

This version correctly simulates cloud component delays by:
1. Tracking when planning requests are sent to cloud
2. Waiting for the appropriate delay before executing planning
3. Actually executing the car's planning and control methods
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

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


class ExtensibleParkingHandler(BaseHandler):
    """
    Parking handler that properly simulates cloud component delays.
    
    The key insight: We can't actually send the car object through the network,
    so we simulate delays by tracking when requests are made and when they should
    be processed.
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
        
        # Delay simulation tracking
        self._cloud_request_sent_frame = -1  # When we sent request to cloud
        self._cloud_request_type = None  # What type of request
        self._waiting_for_cloud = False  # Are we waiting for cloud response
        
        logger.info(f"ExtensibleParkingHandler V2 initialized with cloud config: {cloud_config.name}")
        
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
        
        # Reset delay simulation
        self._cloud_request_sent_frame = -1
        self._cloud_request_type = None
        self._waiting_for_cloud = False
        
        # Create initial observation
        self._update_observation()
    
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
        Process simulation step with proper cloud delay simulation.
        
        Key insight: We simulate delays by tracking when requests are sent
        and only executing planning after the appropriate delay.
        """
        # Execute physics simulation step
        self.step()
        
        # Update dynamic obstacles
        if self.walkers:
            self.dynamic_bbs = update_walkers(self.walkers)
        else:
            self.dynamic_bbs = []
        
        # Update obstacle map every step
        all_bbs = self.static_bbs + self.dynamic_bbs
        self.car.car.obs = union_obstacle_map(
            self.car.car.obs,
            mask_obstacle_map(
                obstacle_map_from_bbs(all_bbs),
                self.car.car.cur.x,
                self.car.car.cur.y
            )
        )
        
        # Check if it's time to plan
        should_plan = self._should_replan()
        
        # Track if we received a cloud response (action is not None means network delivered something)
        cloud_response_received = action is not None and isinstance(action, np.ndarray)
        
        # Handle planning based on cloud configuration
        if self.cloud_config.name == 'baseline':
            # Baseline: Plan immediately when needed
            if should_plan:
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Baseline planning (immediate)")
                
                try:
                    if self.car and self.car.actor:
                        self.car.actor.get_transform()
                        self.car.plan()
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None and len(self.car.car.trajectory) > 0
                        self._planning_active = has_plan
                        
                        if self.frame_idx % 50 == 0:
                            logger.info(f"Baseline planning result: has_plan={has_plan}")
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
                    
        else:
            # Cloud scenarios: Simulate delay
            if should_plan and not self._waiting_for_cloud:
                # Send request to cloud (start waiting)
                self._cloud_request_sent_frame = self.frame_idx
                self._cloud_request_type = self.cloud_config.name
                self._waiting_for_cloud = True
                
                if self.frame_idx % 50 == 0:
                    logger.info(f"Step {self.frame_idx}: Sending {self.cloud_config.name} request to cloud")
            
            # Check if we've waited long enough for cloud response
            if self._waiting_for_cloud and cloud_response_received:
                # Cloud response received - execute planning
                self._waiting_for_cloud = False
                
                if self.frame_idx % 50 == 0:
                    frames_waited = self.frame_idx - self._cloud_request_sent_frame
                    delay_ms = frames_waited * self.scenario_config.timestep * 1000
                    logger.info(f"Step {self.frame_idx}: Cloud response received after {delay_ms:.1f}ms, executing planning")
                
                try:
                    if self.car and self.car.actor:
                        self.car.actor.get_transform()
                        
                        # For all cloud scenarios, we execute the actual planning now
                        # (after the simulated delay)
                        self.car.plan()
                        
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None and len(self.car.car.trajectory) > 0
                        self._planning_active = has_plan
                        
                        if self.frame_idx % 50 == 0:
                            logger.info(f"Cloud-delayed planning result: has_plan={has_plan}")
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
        
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
            logger.info(f"Frame {self.frame_idx}: {self.cloud_config.name}, "
                       f"waiting_for_cloud={self._waiting_for_cloud}, "
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