#!/usr/bin/env python3
"""
Extensible Parking Handler with Cloud Component Integration

This handler supports all three cloud scenarios with proper FogSim network delay handling:
1. Cloud Perception: Raw sensor data → Cloud (delayed) → Local planning → Local control
2. Cloud Planning: Local perception → Cloud planning (delayed) → Local control  
3. Full Cloud: Raw data → Cloud perception → Cloud planning → Cloud control (delayed)
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
    PerceptionData, 
    PlanningData, 
    ControlData,
    CLOUD_SCENARIOS
)
import time

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
class DelayedMessage:
    """Wrapper for messages traveling through network delay."""
    message_type: str  # 'perception', 'planning', 'control'
    data: Union[PerceptionData, PlanningData, ControlData]
    send_time: float
    frame_id: int


class ExtensibleParkingHandler(BaseHandler):
    """
    Extensible parking handler supporting different cloud architectures.
    
    This handler uses FogSim to simulate network delays for cloud components
    while maintaining local processing for on-vehicle components.
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
        
        # Cloud component instances
        self.perception_component, self.planning_component, self.control_component = cloud_config.get_components()
        
        # Network delay state tracking
        self._last_perception_data = None
        self._last_planning_data = None
        self._last_control_data = None
        
        # Messages waiting for network delivery
        self._pending_perception_requests = []
        self._pending_planning_requests = []
        self._pending_control_requests = []
        
        logger.info(f"ExtensibleParkingHandler initialized with cloud config: {cloud_config.name}")
        
    def launch(self) -> None:
        """Launch CARLA and initialize the parking scenario."""
        if self._launched:
            return
            
        try:
            # Initialize CARLA (same as original)
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
        # Same spawning logic as original
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
                    # Just try to destroy - CARLA handles already destroyed actors gracefully
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
        # Cleanup (same as original)
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
            # Clean up any partially created resources
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
        
        # Reset cloud component states
        self._last_perception_data = None
        self._last_planning_data = None
        self._last_control_data = None
        self._pending_perception_requests.clear()
        self._pending_planning_requests.clear()
        self._pending_control_requests.clear()
        
        # Create initial observation
        self._update_observation()
    
    def _process_perception_pipeline(self) -> PerceptionData:
        """Process the perception pipeline based on cloud configuration."""
        # Update dynamic obstacles
        if self.walkers:
            self.dynamic_bbs = update_walkers(self.walkers)
        else:
            self.dynamic_bbs = []
        
        if self.perception_component.location == ComponentLocation.LOCAL:
            # Local perception - process immediately
            return self.perception_component.process(self.car, self.static_bbs, self.dynamic_bbs)
        else:
            # Cloud perception - this should be called when delayed data arrives
            # For now, return the last known perception data if available
            if self._last_perception_data is not None:
                return self._last_perception_data
            else:
                # Fallback: create basic perception data
                transform = self.car.actor.get_transform()
                velocity = self.car.actor.get_velocity()
                return PerceptionData(
                    obstacle_map=np.zeros((100, 100)),  # Empty obstacle map
                    vehicle_position=(transform.location.x, transform.location.y, transform.rotation.yaw),
                    vehicle_velocity=(velocity.x, velocity.y),
                    timestamp=time.time(),
                    frame_id=self.frame_idx
                )
    
    def _process_planning_pipeline(self, perception_data: Optional[PerceptionData]) -> PlanningData:
        """Process the planning pipeline based on cloud configuration."""
        if self.planning_component.location == ComponentLocation.LOCAL:
            # Local planning - process immediately
            if perception_data is not None:
                return self.planning_component.process(perception_data, self.car)
            else:
                # No perception data available - create local perception first
                local_perception = self._process_perception_pipeline()
                return self.planning_component.process(local_perception, self.car)
        else:
            # Cloud planning - use last known planning data if available
            if self._last_planning_data is not None:
                return self._last_planning_data
            else:
                # Fallback: create no-plan data
                return PlanningData(
                    trajectory=[],
                    target_speed=0.0,
                    steering_angle=0.0,
                    has_plan=False,
                    timestamp=time.time(),
                    frame_id=self.frame_idx
                )
    
    def _process_control_pipeline(self, planning_data: PlanningData) -> ControlData:
        """Process the control pipeline based on cloud configuration."""
        if self.control_component.location == ComponentLocation.LOCAL:
            # Local control - process immediately
            return self.control_component.process(planning_data, self.car)
        else:
            # Cloud control - use last known control data if available
            if self._last_control_data is not None:
                # Apply the delayed control commands
                try:
                    import carla
                    control = carla.VehicleControl(
                        throttle=self._last_control_data.throttle,
                        brake=self._last_control_data.brake,
                        steer=self._last_control_data.steer
                    )
                    if self.car and self.car.actor:
                        try:
                            self.car.actor.apply_control(control)
                        except RuntimeError:
                            logger.error("Car actor is destroyed or None, cannot apply control")
                            self._episode_done = True
                    else:
                        logger.error("Car actor is destroyed or None, cannot apply control")
                        self._episode_done = True
                except Exception as e:
                    logger.warning(f"Failed to apply delayed control: {e}")
                    # Set emergency stop if control fails
                    try:
                        if self.car and self.car.actor:
                            try:
                                emergency_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                                self.car.actor.apply_control(emergency_control)
                            except RuntimeError:
                                pass
                    except:
                        pass
                
                return self._last_control_data
            else:
                # Fallback: create stop command
                control_data = ControlData(
                    throttle=0.0,
                    brake=0.5,
                    steer=0.0,
                    timestamp=time.time(),
                    frame_id=self.frame_idx
                )
                # Apply stop command with improved error handling
                try:
                    import carla
                    control = carla.VehicleControl(throttle=0.0, brake=0.5, steer=0.0)
                    if self.car and self.car.actor:
                        try:
                            self.car.actor.apply_control(control)
                        except RuntimeError:
                            logger.error("Car actor is destroyed or None, cannot apply stop control")
                            self._episode_done = True
                    else:
                        logger.error("Car actor is destroyed or None, cannot apply stop control")
                        self._episode_done = True
                except Exception as e:
                    logger.warning(f"Failed to apply stop control: {e}")
                    self._episode_done = True
                
                return control_data
    
    def _should_replan(self):
        """Determine if replanning is needed with error handling."""
        try:
            if self.car is None or not hasattr(self.car, 'actor') or self.car.actor is None:
                return False
            # Try to access car data to check if it's still valid
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
        """Update the observation vector with error handling."""
        if self.car is None:
            self._observation = np.zeros(15)
            return
            
        try:
            # Check if car actor is still valid
            if not hasattr(self.car, 'actor') or self.car.actor is None:
                logger.error("Car actor is None, setting default observation")
                self._observation = np.zeros(15)
                self._episode_done = True
                return
            
            # Try to access car data to check if it's still valid
            try:
                self.car.actor.get_transform()
            except RuntimeError:
                logger.error("Car actor is destroyed, setting default observation")
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
                
            # Create BASE observation vector (15 elements)
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
                # Obstacle map features
                np.sum(self.car.car.obs.obs) / (self.car.car.obs.obs.size + 1e-6),
                np.mean(self.car.car.obs.obs),
                np.std(self.car.car.obs.obs),
                # Parking spot location
                self.car.car.destination.x,
                self.car.car.destination.y,
                self.car.car.destination.angle,
            ])
            
        except Exception as e:
            logger.error(f"Failed to update observation at frame {self.frame_idx}: {e}")
            # Provide a safe fallback observation
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
        
        # Tick simulation with error handling
        try:
            self.world.tick()
        except Exception as e:
            logger.error(f"World tick failed at frame {self.frame_idx}: {e}")
            self._episode_done = True
            return
            
        try:
            self.car.localize()
        except Exception as e:
            logger.error(f"Car localization failed at frame {self.frame_idx}: {e}")
            # Continue execution but log the error
        
        # Process recording frames if video recording is enabled
        if self.recording_file:
            latency_ms = self.scenario_config.network_delay * 1000.0
            source_rate_kbps = self.scenario_config.source_rate / 1000.0
            # Create concise cloud mode text for video overlay
            cloud_mode_short = {
                "baseline": "Baseline (All Local)",
                "cloud_perception": "Cloud Perception",
                "cloud_planning": "Cloud Planning", 
                "full_cloud": "Full Cloud"
            }.get(self.cloud_config.name, self.cloud_config.name)
            cloud_mode = cloud_mode_short
            self.car.process_recording_frames(latency_ms, source_rate_kbps, cloud_mode)
        
        # Increment frame counter
        self.frame_idx += 1
                
    def step_with_action(self, action: Optional[np.ndarray]) -> Tuple:
        """
        Process simulation step with proper cloud message handling.
        
        For cloud scenarios:
        - action contains delayed messages from cloud processing
        - We need to send appropriate data through the network for cloud components
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
        
        # Track if we received a delayed response from cloud
        received_cloud_response = action is not None and isinstance(action, np.ndarray)
        
        # For cloud scenarios, if we receive a delayed response, process it and plan
        if received_cloud_response and self.cloud_config.name != 'baseline':
            if self.frame_idx % 50 == 0:
                print(f"Step {self.frame_idx}: Received cloud response, will plan now")
            
            # For cloud perception: plan with the delayed perception data
            if self.cloud_config.perception_location == ComponentLocation.CLOUD:
                try:
                    if self.car and self.car.actor:
                        # Check if actor is valid by accessing transform
                        self.car.actor.get_transform()
                        self.car.plan()
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None and len(self.car.car.trajectory) > 0
                        self._planning_active = has_plan
                        
                        if self.frame_idx % 50 == 0:
                            print(f"Planning result (after cloud perception): has_plan={has_plan}, path_length={len(self.car.car.trajectory) if has_plan else 0}")
                    else:
                        logger.error("Car actor is None or destroyed, cannot plan after cloud perception")
                        self._planning_active = False
                        self._episode_done = True
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
                    
            # For cloud planning: use the delayed planning result
            elif self.cloud_config.planning_location == ComponentLocation.CLOUD:
                self._planning_active = True  # Assume cloud sent valid plan
                if self.frame_idx % 50 == 0:
                    print(f"Using cloud planning result")
        
        # For baseline or when it's a regular planning interval without cloud
        should_plan = (self.frame_idx % self.scenario_config.replan_interval == 0)
        
        if should_plan and not received_cloud_response:
            if self.frame_idx % 50 == 0:
                print(f"Step {self.frame_idx}: {self.cloud_config.name} planning triggered")
                print(f"Planning: pos=({self.car.car.cur.x:.1f},{self.car.car.cur.y:.1f}) dest=({self.car.car.destination.x:.1f},{self.car.car.destination.y:.1f})")
            
            # For baseline (all local): plan immediately
            if self.cloud_config.name == 'baseline':
                try:
                    if self.car and self.car.actor:
                        try:
                            # Check if actor is valid by accessing transform
                            self.car.actor.get_transform()
                            self.car.plan()
                        except RuntimeError:
                            logger.error("Car actor is destroyed, cannot plan")
                            self._planning_active = False
                            self._episode_done = True
                            return
                        has_plan = hasattr(self.car.car, 'trajectory') and self.car.car.trajectory is not None and len(self.car.car.trajectory) > 0
                        self._planning_active = has_plan
                        
                        if self.frame_idx % 50 == 0:
                            print(f"Planning result (baseline): has_plan={has_plan}, path_length={len(self.car.car.trajectory) if has_plan else 0}")
                    else:
                        logger.error("Car actor is destroyed, cannot plan")
                        self._planning_active = False
                        self._episode_done = True
                except Exception as e:
                    logger.error(f"Planning failed: {e}")
                    self._planning_active = False
            # For cloud scenarios, we'll wait for the delayed response
        
        # Execute control with error handling
        if self._planning_active:
            try:
                if self.car and self.car.actor:
                    try:
                        self.car.run_step()
                    except RuntimeError:
                        logger.error("Car actor is destroyed, stopping episode")
                        self._episode_done = True
                        self._planning_active = False
                else:
                    logger.error("Car actor is destroyed, stopping episode")
                    self._episode_done = True
                    self._planning_active = False
            except Exception as e:
                logger.error(f"Car run_step failed at frame {self.frame_idx}: {e}")
                self._planning_active = False
                # Try emergency stop
                try:
                    if self.car and self.car.actor:
                        try:
                            import carla
                            emergency_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                            self.car.actor.apply_control(emergency_control)
                        except RuntimeError:
                            pass
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
            delayed_action_received = action is not None
            print(f"FogSim Debug - Frame {self.frame_idx}: {self.cloud_config.name}, delayed_action={delayed_action_received}, planning_active={self._planning_active}")
        
        # Additional debug info every few frames
        if self.frame_idx % 200 == 0:
            print(f"\n=== DETAILED DEBUG Frame {self.frame_idx} ===")
            print(f"Cloud config: {self.cloud_config.name}")
            print(f"Delayed action received: {action is not None}")
            print(f"Planning active: {self._planning_active}")
            print(f"Vehicle position: ({self.car.car.cur.x:.1f}, {self.car.car.cur.y:.1f})")
            print(f"==========================================\n")
        
        return observation, reward, success, terminated, truncated, states
        
    def render(self) -> Optional[np.ndarray]:
        """Render is not implemented for this handler."""
        return None
        
    def close(self) -> None:
        """Clean up resources with comprehensive error handling."""
        logger.info("Cleaning up CARLA resources...")
        
        # Clean up recording camera
        if self.recording_cam is not None:
            try:
                self.recording_cam.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy recording camera: {e}")
            finally:
                self.recording_cam = None
        
        # Clean up car
        if self.car is not None:
            try:
                if hasattr(self.car, 'actor') and self.car.actor:
                    self.car.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy car: {e}")
            finally:
                self.car = None
        
        # Clean up other actors
        try:
            self._cleanup_actors()
        except Exception as e:
            logger.warning(f"Failed to clean up actors: {e}")
        
        # Final world tick
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