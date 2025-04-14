import logging
import numpy as np
import time
from typing import Any, Dict, Optional, Tuple
from ..base import BaseCoSimulator
from .client_adapter import VisualizationClientAdapter

# Set up logging
logger = logging.getLogger(__name__)

class VisualizationCoSimulator:
    """Wrapper class that extends co-simulators with visualization capabilities."""
    
    def __init__(self, co_simulator: BaseCoSimulator, server_url='http://localhost:5000', 
                 simulation_id=None, auto_connect=True):
        """
        Initialize the visualization wrapper.
        
        Args:
            co_simulator: The co-simulator to wrap
            server_url: URL of the visualization server
            simulation_id: Unique identifier for this simulation
            auto_connect: Whether to automatically connect to the server
        """
        self.co_simulator = co_simulator
        
        # Detect simulator type
        if 'GymCoSimulator' in co_simulator.__class__.__name__:
            simulator_type = 'gym'
        elif 'CarlaCoSimulator' in co_simulator.__class__.__name__:
            simulator_type = 'carla'
        else:
            simulator_type = co_simulator.__class__.__name__
        
        # Create the visualization client
        self.viz_client = VisualizationClientAdapter(
            server_url=server_url,
            simulation_id=simulation_id,
            simulator_type=simulator_type
        )
        
        # Metrics storage
        self.latest_metrics = {}
        self.last_step_time = time.time()
        self.last_action = None
        self.step_count = 0
        
        # Track whether to send visualizations
        self.render_enabled = True
        
        # Register command handlers
        self._register_command_handlers()
        
        # Connect to the visualization server if requested
        if auto_connect:
            self.viz_client.connect()
        
        logger.info(f"VisualizationCoSimulator initialized for {simulator_type}")
    
    def _register_command_handlers(self):
        """Register command handlers for the visualization client."""
        # Handler for enabling/disabling rendering
        self.viz_client.register_command_handler('set_render_enabled', self._handle_set_render_enabled)
        
        # Handler for changing network parameters
        self.viz_client.register_command_handler('set_network_params', self._handle_set_network_params)
        
        # Handler for resetting the simulation
        self.viz_client.register_command_handler('reset', self._handle_reset)
    
    def _handle_set_render_enabled(self, params):
        """Handle command to enable/disable rendering."""
        enabled = params.get('enabled', True)
        self.render_enabled = enabled
        logger.info(f"Rendering {'enabled' if enabled else 'disabled'}")
        return {'render_enabled': self.render_enabled}
    
    def _handle_set_network_params(self, params):
        """Handle command to change network parameters."""
        # This requires access to the network simulator
        if hasattr(self.co_simulator, 'network_simulator'):
            network_sim = self.co_simulator.network_simulator
            
            # Attempt to set each provided parameter
            results = {}
            
            if 'latency' in params:
                latency = float(params['latency'])
                # Implementation depends on the specific network simulator API
                try:
                    # This is an example - actual implementation will depend on network simulator
                    if hasattr(network_sim, 'set_latency'):
                        network_sim.set_latency(latency)
                        results['latency'] = latency
                    else:
                        logger.warning("Network simulator does not support set_latency")
                except Exception as e:
                    logger.error(f"Failed to set latency: {str(e)}")
                    results['latency_error'] = str(e)
            
            if 'packet_loss' in params:
                packet_loss = float(params['packet_loss'])
                try:
                    if hasattr(network_sim, 'set_packet_loss'):
                        network_sim.set_packet_loss(packet_loss)
                        results['packet_loss'] = packet_loss
                    else:
                        logger.warning("Network simulator does not support set_packet_loss")
                except Exception as e:
                    logger.error(f"Failed to set packet loss: {str(e)}")
                    results['packet_loss_error'] = str(e)
            
            if 'bandwidth' in params:
                bandwidth = float(params['bandwidth'])
                try:
                    if hasattr(network_sim, 'set_bandwidth'):
                        network_sim.set_bandwidth(bandwidth)
                        results['bandwidth'] = bandwidth
                    else:
                        logger.warning("Network simulator does not support set_bandwidth")
                except Exception as e:
                    logger.error(f"Failed to set bandwidth: {str(e)}")
                    results['bandwidth_error'] = str(e)
            
            return results
        else:
            logger.warning("Co-simulator does not have a network_simulator attribute")
            return {'error': 'No network simulator available'}
    
    def _handle_reset(self, params):
        """Handle command to reset the simulation."""
        try:
            self.reset()
            return {'reset': 'success'}
        except Exception as e:
            logger.error(f"Reset failed: {str(e)}")
            return {'reset_error': str(e)}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation with visualization.
        
        Args:
            action: Action to be taken in the robotics simulator
            
        Returns:
            Tuple containing observation, reward, done, and info
        """
        # Record the action for metrics
        self.last_action = action
        
        # Record step start time
        step_start_time = time.time()
        
        # Execute the step in the co-simulator
        result = self.co_simulator.step(action)
        
        # Unpack the result based on its structure
        if len(result) == 4:
            observation, reward, done, info = result
        else:
            # Handle any other result structures if needed
            observation = result
            reward, done, info = 0.0, False, {}
        
        # Calculate step duration
        step_duration = time.time() - step_start_time
        
        # Update metrics
        self.step_count += 1
        fps = 1.0 / step_duration if step_duration > 0 else 0
        
        self.latest_metrics = {
            'step_count': self.step_count,
            'step_duration': step_duration,
            'fps': fps,
            'reward': reward,
            'done': done
        }
        
        # Include any latency metrics from the info dict
        if isinstance(info, dict):
            for key, value in info.items():
                if 'latency' in key:
                    self.latest_metrics[key] = value
        
        # Send metrics to visualization server
        self.viz_client.send_metrics(self.latest_metrics)
        
        # If rendering is enabled, render and send the frame
        if self.render_enabled:
            try:
                frame = self.render(mode='rgb_array')
                if frame is not None:
                    self.viz_client.send_frame(frame)
            except Exception as e:
                logger.warning(f"Render failed: {str(e)}")
        
        return result
    
    def reset(self) -> np.ndarray:
        """Reset the co-simulator and visualization state."""
        # Reset the co-simulator
        observation = self.co_simulator.reset()
        
        # Reset metrics
        self.latest_metrics = {}
        self.last_step_time = time.time()
        self.last_action = None
        self.step_count = 0
        
        # Send initial frame if rendering is enabled
        if self.render_enabled:
            try:
                frame = self.render(mode='rgb_array')
                if frame is not None:
                    self.viz_client.send_frame(frame)
            except Exception as e:
                logger.warning(f"Initial render failed: {str(e)}")
        
        return observation
    
    def render(self, mode: str = 'human') -> Any:
        """Render the current state of the simulation."""
        return self.co_simulator.render(mode=mode)
    
    def close(self) -> None:
        """Clean up resources."""
        # Disconnect from visualization server
        self.viz_client.disconnect()
        
        # Close the co-simulator
        self.co_simulator.close()
    
    # Proxy all other attributes to the wrapped co-simulator
    def __getattr__(self, name):
        """Forward any other attribute access to the wrapped co-simulator."""
        return getattr(self.co_simulator, name) 