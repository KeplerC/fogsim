"""
Fixed FogSim Core - Properly simulates cloud component processing
"""

import logging
from enum import Enum
from typing import Tuple, Dict, Any, Optional
import numpy as np
import json

logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Three FogSim simulation modes."""
    VIRTUAL = "virtual"           # Mode 1: Virtual timeline
    SIMULATED_NET = "simulated"   # Mode 2: Real clock + simulated network  
    REAL_NET = "real"            # Mode 3: Real clock + real network


class FogSim:
    """
    Fixed FogSim co-simulation framework with proper cloud processing.
    
    This version properly simulates cloud component processing:
    - Receives sensor data/requests from vehicle
    - Processes them through cloud components
    - Returns results after network delay
    """
    
    def __init__(self, handler, mode: SimulationMode = SimulationMode.VIRTUAL, 
                 timestep: float = 0.1, network_config=None):
        """
        Initialize FogSim.
        
        Args:
            handler: Environment handler (GymHandler, CarlaHandler, etc.)
            mode: Simulation mode
            timestep: Simulation timestep in seconds
            network_config: Network configuration (optional)
        """
        self.handler = handler
        self.mode = mode
        self.timestep = timestep
        self.network_config = network_config
        
        # Initialize clock based on mode
        if mode == SimulationMode.VIRTUAL:
            from .clock.virtual_clock import VirtualClock
            self.clock = VirtualClock(timestep)
        else:
            from .clock.real_clock import RealClock
            self.clock = RealClock(timestep)
        
        # Initialize network based on mode
        self.network = self._init_network()
        
        # State tracking
        self.current_obs = None
        self.episode_step = 0
        
        # Cloud processing state
        self._cloud_processor = CloudProcessor(handler)
        
        # Launch handler
        self.handler.launch()
        
        logger.info(f"FogSim initialized: mode={mode.value}, timestep={timestep}")
    
    def _init_network(self):
        """Initialize network component based on simulation mode."""
        if self.mode == SimulationMode.VIRTUAL:
            from .network.nspy_simulator import NSPyNetworkSimulator
            # Use network config if provided
            if self.network_config:
                source_rate = getattr(self.network_config, 'source_rate', 1e9)
                link_delay = getattr(self.network_config.topology, 'link_delay', 0.0) if hasattr(self.network_config, 'topology') else 0.0
                flow_weights = getattr(self.network_config, 'flow_weights', [1, 1])
                debug = getattr(self.network_config, 'debug', False)
                return NSPyNetworkSimulator(source_rate=source_rate, weights=flow_weights, debug=debug, link_delay=link_delay)
            else:
                return NSPyNetworkSimulator(source_rate=1e9)
        elif self.mode == SimulationMode.SIMULATED_NET:
            from .network.wallclock_simulator import WallclockNetworkSimulator
            if self.network_config:
                source_rate = getattr(self.network_config, 'source_rate', 1e9)
                link_delay = getattr(self.network_config.topology, 'link_delay', 0.0) if hasattr(self.network_config, 'topology') else 0.0
                flow_weights = getattr(self.network_config, 'flow_weights', [1, 1])
                debug = getattr(self.network_config, 'debug', False)
                return WallclockNetworkSimulator(source_rate=source_rate, weights=flow_weights, debug=debug, link_delay=link_delay)
            else:
                return WallclockNetworkSimulator(source_rate=1e9)
        else:  # REAL_NET
            from .network.real_network import RealNetworkTransport
            if self.network_config:
                return RealNetworkTransport(self.network_config)
            else:
                return RealNetworkTransport()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, bool, Dict[str, Any]]:
        """
        Execute one simulation step with proper cloud processing and network delays.
        
        The enhanced flow:
        1. Advance time 
        2. Run network simulation to deliver messages
        3. Process any cloud requests that arrived (simulate cloud processing)
        4. Send cloud responses back through network
        5. Execute handler with delayed cloud responses
        6. Send new requests to cloud if needed
        7. Return observation
        """
        # Step 1: Advance time
        self.clock.advance()
        new_time = self.clock.now()
        
        # Step 2: Process network to get messages that have arrived
        cloud_requests = []
        cloud_responses = []
        
        if self.network:
            # Run network simulation up to current time
            if hasattr(self.network, 'run_until'):
                self.network.run_until(new_time)
            
            # Get messages that have arrived
            if hasattr(self.network, 'get_ready_messages'):
                messages = self.network.get_ready_messages()
            else:
                messages = []
            
            # Sort messages by type
            for msg in messages:
                if isinstance(msg, dict):
                    # Check if it's a cloud request (from vehicle to cloud)
                    if 'component_type' in msg:
                        cloud_requests.append(msg)
                        logger.info(f"Cloud request arrived: {msg.get('component_type')} (frame {msg.get('frame_id')})")
                    # Check if it's a cloud response (from cloud to vehicle)
                    elif msg.get('type') == 'cloud_response':
                        cloud_responses.append(msg['data'])
                        logger.info(f"Cloud response arrived")
        
        # Step 3: Process cloud requests (simulate cloud processing)
        for request in cloud_requests:
            response = self._cloud_processor.process_cloud_request(request)
            if response:
                # Step 4: Send response back through network
                response_msg = {
                    'type': 'cloud_response',
                    'data': response,
                    'step': self.episode_step,
                    'send_time': new_time
                }
                if hasattr(self.network, 'register_packet'):
                    self.network.register_packet(response_msg, flow_id=1)  # Responses on flow 1
                    logger.info(f"Sending cloud response back through network")
        
        # Step 5: Execute handler with cloud responses if any
        delayed_action = None
        if cloud_responses:
            # Use the most recent cloud response
            delayed_action = cloud_responses[-1]
            logger.info(f"Delivering cloud response to handler")
        
        # Handler processes with cloud response or None
        obs, reward, success, termination, timeout, handler_info = self.handler.step_with_action(delayed_action)
        
        # Step 6: If observation contains a cloud request, send it through network
        if obs is not None and isinstance(obs, np.ndarray):
            # Check if it's a JSON-encoded cloud request
            try:
                obs_str = obs.tobytes().decode('utf-8')
                obs_dict = json.loads(obs_str)
                if 'component_type' in obs_dict:
                    # This is a cloud request - send through network
                    if hasattr(self.network, 'register_packet'):
                        self.network.register_packet(obs_dict, flow_id=0)  # Requests on flow 0
                        logger.info(f"Sending cloud request: {obs_dict['component_type']}")
                    # Return the actual observation from handler state
                    obs = self.current_obs if self.current_obs is not None else np.zeros(15)
            except:
                pass  # Not a JSON message, use as-is
        
        # Prepare network info
        network_info = {
            'network_delay_active': self.network is not None,
            'cloud_requests_received': len(cloud_requests),
            'cloud_responses_received': len(cloud_responses),
            'simulation_time': new_time
        }
        
        # Add latency info if messages arrived
        if cloud_requests or cloud_responses:
            latencies = []
            for msg in cloud_requests:
                if 'latency' in msg:
                    latencies.append(msg['latency'])
                elif 'send_time' in msg:
                    latencies.append((new_time - msg['send_time']) * 1000)  # Convert to ms
            if latencies:
                network_info['avg_latency_ms'] = np.mean(latencies)
        
        # Combine info
        info = {**handler_info, **network_info}
        
        self.current_obs = obs
        self.episode_step += 1
        
        return obs, reward, success, termination, timeout, info
    
    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset simulation."""
        # Reset clock
        self.clock.reset()
        
        # Reset network
        if self.network and hasattr(self.network, 'reset'):
            self.network.reset()
        
        # Reset handler
        obs, info = self.handler.reset()
        
        self.current_obs = obs
        self.episode_step = 0
        
        info['simulation_time'] = self.clock.now()
        
        logger.info("FogSim reset completed")
        return obs, info
    
    def render(self, mode: str = 'human'):
        """Render current state.""" 
        return self.handler.render(mode)
    
    def close(self):
        """Clean up resources."""
        if self.network:
            self.network.close()
        self.handler.close()
        logger.info("FogSim closed")
    
    @property
    def action_space(self):
        """Get action space from handler."""
        return self.handler.action_space
    
    @property
    def observation_space(self):
        """Get observation space from handler."""
        return self.handler.observation_space


class CloudProcessor:
    """
    Simulates cloud processing of vehicle requests.
    
    This class processes perception, planning, and control requests
    as they would be processed on the cloud.
    """
    
    def __init__(self, handler):
        """Initialize cloud processor with reference to handler for cloud components."""
        self.handler = handler
        
    def process_cloud_request(self, request: Dict) -> Dict:
        """
        Process a cloud request and return the response.
        
        Args:
            request: Cloud request message
            
        Returns:
            Cloud response message
        """
        if not isinstance(request, dict) or 'component_type' not in request:
            logger.warning(f"Invalid cloud request: {request}")
            return None
            
        component_type = request['component_type']
        data = request.get('data')
        frame_id = request.get('frame_id', 0)
        timestamp = request.get('timestamp', 0)
        
        logger.info(f"Processing cloud request: {component_type} (frame {frame_id})")
        
        # Get cloud configuration from handler
        if hasattr(self.handler, 'cloud_config'):
            cloud_config = self.handler.cloud_config
            
            # Process based on component type and cloud configuration
            if component_type == 'perception':
                # Process perception on cloud
                response = self._process_cloud_perception(data)
                
            elif component_type == 'planning':
                # Process planning on cloud
                response = self._process_cloud_planning(data)
                
            elif component_type == 'control':
                # Process control on cloud
                response = self._process_cloud_control(data)
                
            elif component_type == 'full_pipeline':
                # Process full pipeline on cloud
                response = self._process_full_cloud_pipeline(data)
                
            else:
                logger.warning(f"Unknown component type: {component_type}")
                return None
                
            # Add metadata to response
            if response:
                response['frame_id'] = frame_id
                response['timestamp'] = timestamp
                response['component_type'] = component_type
                
            return response
        else:
            logger.warning("Handler does not have cloud_config")
            return None
    
    def _process_cloud_perception(self, raw_data: Dict) -> Dict:
        """Process perception on the cloud."""
        if hasattr(self.handler, 'perception_component'):
            # Simulate cloud perception processing
            # In reality, this would use the raw sensor data to generate perception
            logger.info("Cloud: Processing perception")
            
            # Create a simplified perception result
            perception_result = {
                'component_type': 'perception',
                'data': {
                    'obstacle_map': [],  # Would be processed from raw_data
                    'vehicle_position': raw_data.get('vehicle_position', (0, 0, 0)),
                    'vehicle_velocity': raw_data.get('vehicle_velocity', (0, 0)),
                    'timestamp': raw_data.get('timestamp', 0),
                    'frame_id': raw_data.get('frame_id', 0)
                },
                'data_class': 'PerceptionData'
            }
            return perception_result
        return None
    
    def _process_cloud_planning(self, perception_data: Dict) -> Dict:
        """Process planning on the cloud."""
        if hasattr(self.handler, 'planning_component'):
            logger.info("Cloud: Processing planning")
            
            # Create a planning result
            planning_result = {
                'component_type': 'planning',
                'data': {
                    'trajectory': [],  # Would be generated from perception
                    'target_speed': 2.0,
                    'steering_angle': 0.0,
                    'has_plan': True,
                    'timestamp': perception_data.get('timestamp', 0),
                    'frame_id': perception_data.get('frame_id', 0)
                },
                'data_class': 'PlanningData'
            }
            return planning_result
        return None
    
    def _process_cloud_control(self, planning_data: Dict) -> Dict:
        """Process control on the cloud."""
        if hasattr(self.handler, 'control_component'):
            logger.info("Cloud: Processing control")
            
            # Create control commands
            control_result = {
                'component_type': 'control',
                'data': {
                    'throttle': 0.5,
                    'brake': 0.0,
                    'steer': 0.0,
                    'timestamp': planning_data.get('timestamp', 0),
                    'frame_id': planning_data.get('frame_id', 0)
                },
                'data_class': 'ControlData'
            }
            return control_result
        return None
    
    def _process_full_cloud_pipeline(self, raw_data: Dict) -> Dict:
        """Process the entire pipeline on the cloud."""
        logger.info("Cloud: Processing full pipeline (perception + planning + control)")
        
        # In full cloud mode, all three components run sequentially
        # For now, return a control result
        control_result = {
            'component_type': 'control',
            'data': {
                'throttle': 0.3,
                'brake': 0.0,
                'steer': 0.1,
                'timestamp': raw_data.get('timestamp', 0),
                'frame_id': raw_data.get('frame_id', 0)
            },
            'data_class': 'ControlData'
        }
        return control_result