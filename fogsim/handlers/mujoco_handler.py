"""Mujoco simulator handler for FogSim.

This module provides a handler implementation for Mujoco physics simulator,
wrapping it to work with the FogSim framework.
"""

from typing import Any, Dict, Optional, Union, Tuple
import numpy as np
import logging
import os

from .base_handler import BaseHandler

mujoco = None
mujoco_py = None
legacy_mujoco = False

# Try modern mujoco first
try:
    import mujoco
except ImportError:
    # Only try legacy mujoco-py if modern mujoco is not available
    try:
        import mujoco_py
        legacy_mujoco = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class MujocoHandler(BaseHandler):
    """Handler for Mujoco physics simulator.
    
    This handler supports both the new Mujoco Python bindings (mujoco) and
    the legacy mujoco-py library, providing a consistent interface for
    simulation control.
    
    Args:
        model_path: Path to the Mujoco XML model file
        model_xml: XML string defining the model (if model_path not provided)
        render_mode: Rendering mode ('human', 'rgb_array', 'depth_array', None)
        render_width: Width of rendered images (default: 640)
        render_height: Height of rendered images (default: 480)
        camera_name: Name of camera to render from (optional)
        use_legacy: Force use of legacy mujoco-py (default: auto-detect)
    """
    
    def __init__(self,
                 model_path: Optional[str] = None,
                 model_xml: Optional[str] = None,
                 render_mode: Optional[str] = None,
                 render_width: int = 640,
                 render_height: int = 480,
                 camera_name: Optional[str] = None,
                 use_legacy: Optional[bool] = None):
        """Initialize the Mujoco handler."""
        if model_path is None and model_xml is None:
            raise ValueError("Either model_path or model_xml must be provided")
        
        if model_path is not None and model_xml is not None:
            raise ValueError("Only one of model_path or model_xml should be provided")
        
        # Auto-detect which Mujoco to use
        if use_legacy is None:
            use_legacy = legacy_mujoco and mujoco is None
        
        self.use_legacy = use_legacy
        
        if self.use_legacy and mujoco_py is None:
            raise ImportError(
                "mujoco-py is not installed. Please install it with: "
                "pip install mujoco-py"
            )
        elif not self.use_legacy and mujoco is None:
            raise ImportError(
                "Mujoco is not installed. Please install it with: "
                "pip install mujoco"
            )
        
        self.model_path = model_path
        self.model_xml = model_xml
        self.render_mode = render_mode
        self.render_width = render_width
        self.render_height = render_height
        self.camera_name = camera_name
        
        self._model = None
        self._sim = None
        self._viewer = None
        self._renderer = None
        self._launched = False
        self._step_count = 0
        self._episode_count = 0
        self._initial_state = None
    
    def launch(self) -> None:
        """Launch the Mujoco simulator."""
        if self._launched:
            logger.warning("Mujoco handler already launched")
            return
        
        try:
            if self.use_legacy:
                self._launch_legacy()
            else:
                self._launch_modern()
            
            # Store initial state for reset
            self._initial_state = self._get_state()
            
            self._launched = True
            logger.info("Mujoco handler launched successfully")
            
        except Exception as e:
            logger.error(f"Failed to launch Mujoco: {e}")
            raise
    
    def _launch_legacy(self) -> None:
        """Launch using legacy mujoco-py."""
        if self.model_path:
            self._model = mujoco_py.load_model_from_path(self.model_path)
        else:
            self._model = mujoco_py.load_model_from_xml(self.model_xml)
        
        self._sim = mujoco_py.MjSim(self._model)
        
        if self.render_mode == 'human':
            self._viewer = mujoco_py.MjViewer(self._sim)
        elif self.render_mode in ['rgb_array', 'depth_array']:
            self._renderer = mujoco_py.MjRenderContextOffscreen(
                self._sim, device_id=-1
            )
    
    def _launch_modern(self) -> None:
        """Launch using modern mujoco Python bindings."""
        if self.model_path:
            self._model = mujoco.MjModel.from_xml_path(self.model_path)
        else:
            self._model = mujoco.MjModel.from_xml_string(self.model_xml)
        
        self._data = mujoco.MjData(self._model)
        
        if self.render_mode in ['human', 'rgb_array', 'depth_array']:
            self._renderer = mujoco.Renderer(self._model)
    
    def set_states(self, states: Optional[Dict[str, Any]] = None,
                   action: Optional[np.ndarray] = None) -> None:
        """Set simulator states.
        
        Args:
            states: Optional state dictionary containing:
                - 'qpos': Joint positions
                - 'qvel': Joint velocities
                - 'ctrl': Control inputs
            action: Control action to apply
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if states is None:
            # Reset to initial state
            self._set_state(self._initial_state)
            self._step_count = 0
            self._episode_count += 1
        else:
            # Set specific states
            if self.use_legacy:
                if 'qpos' in states:
                    self._sim.data.qpos[:] = states['qpos']
                if 'qvel' in states:
                    self._sim.data.qvel[:] = states['qvel']
                if 'ctrl' in states:
                    self._sim.data.ctrl[:] = states['ctrl']
                self._sim.forward()
            else:
                if 'qpos' in states:
                    self._data.qpos[:] = states['qpos']
                if 'qvel' in states:
                    self._data.qvel[:] = states['qvel']
                if 'ctrl' in states:
                    self._data.ctrl[:] = states['ctrl']
                mujoco.mj_forward(self._model, self._data)
        
        if action is not None:
            # Apply control action
            if self.use_legacy:
                self._sim.data.ctrl[:] = action
            else:
                self._data.ctrl[:] = action
    
    def _get_state(self) -> Dict[str, np.ndarray]:
        """Get current state as a dictionary."""
        if self.use_legacy:
            return {
                'qpos': self._sim.data.qpos.copy(),
                'qvel': self._sim.data.qvel.copy(),
                'ctrl': self._sim.data.ctrl.copy()
            }
        else:
            return {
                'qpos': self._data.qpos.copy(),
                'qvel': self._data.qvel.copy(),
                'ctrl': self._data.ctrl.copy()
            }
    
    def _set_state(self, state: Dict[str, np.ndarray]) -> None:
        """Set state from a dictionary."""
        if self.use_legacy:
            self._sim.data.qpos[:] = state['qpos']
            self._sim.data.qvel[:] = state['qvel']
            self._sim.data.ctrl[:] = state['ctrl']
            self._sim.forward()
        else:
            self._data.qpos[:] = state['qpos']
            self._data.qvel[:] = state['qvel']
            self._data.ctrl[:] = state['ctrl']
            mujoco.mj_forward(self._model, self._data)
    
    def get_states(self) -> Dict[str, Any]:
        """Get current simulator states.
        
        Returns:
            Dictionary containing state information
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.use_legacy:
            observation = np.concatenate([
                self._sim.data.qpos.flat,
                self._sim.data.qvel.flat
            ])
            
            return {
                'observation': observation,
                'qpos': self._sim.data.qpos.copy(),
                'qvel': self._sim.data.qvel.copy(),
                'ctrl': self._sim.data.ctrl.copy(),
                'time': self._sim.data.time,
                'step_count': self._step_count,
                'episode_count': self._episode_count
            }
        else:
            observation = np.concatenate([
                self._data.qpos.flat,
                self._data.qvel.flat
            ])
            
            return {
                'observation': observation,
                'qpos': self._data.qpos.copy(),
                'qvel': self._data.qvel.copy(),
                'ctrl': self._data.ctrl.copy(),
                'time': self._data.time,
                'step_count': self._step_count,
                'episode_count': self._episode_count
            }
    
    def step(self) -> None:
        """Step the simulation forward."""
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.use_legacy:
            self._sim.step()
        else:
            mujoco.mj_step(self._model, self._data)
        
        self._step_count += 1
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state.
        
        Returns:
            Rendered image as numpy array or None
        """
        if not self._launched:
            raise RuntimeError("Handler not launched. Call launch() first.")
        
        if self.render_mode is None:
            return None
        
        if self.use_legacy:
            if self.render_mode == 'human' and self._viewer is not None:
                self._viewer.render()
                return None
            elif self.render_mode == 'rgb_array' and self._renderer is not None:
                self._renderer.render(self.render_width, self.render_height)
                data = self._renderer.read_pixels(
                    self.render_width, self.render_height, depth=False
                )
                return data[::-1, :, :]  # Flip vertically
            elif self.render_mode == 'depth_array' and self._renderer is not None:
                self._renderer.render(self.render_width, self.render_height)
                data = self._renderer.read_pixels(
                    self.render_width, self.render_height, depth=True
                )[1]
                return data[::-1, :]  # Flip vertically
        else:
            if self._renderer is not None:
                self._renderer.update_scene(self._data, self.camera_name)
                
                if self.render_mode == 'human':
                    # For human mode, we still return the array
                    # User can display it using their preferred method
                    return self._renderer.render()
                elif self.render_mode == 'rgb_array':
                    return self._renderer.render()
                elif self.render_mode == 'depth_array':
                    # Modern mujoco doesn't have direct depth rendering
                    # Return regular render for now
                    return self._renderer.render()
        
        return None
    
    def close(self) -> None:
        """Clean up Mujoco resources."""
        if self._launched:
            if self.use_legacy:
                if self._viewer is not None:
                    self._viewer = None
                if self._renderer is not None:
                    self._renderer = None
            else:
                if self._renderer is not None:
                    self._renderer = None
            
            self._launched = False
            logger.info("Mujoco handler closed")
    
    def get_extra(self) -> Dict[str, Any]:
        """Get extra metadata.
        
        Returns:
            Dictionary containing Mujoco-specific information
        """
        if not self._launched:
            return {
                'model_path': self.model_path,
                'render_mode': self.render_mode,
                'launched': False,
                'use_legacy': self.use_legacy
            }
        
        if self.use_legacy:
            return {
                'model_path': self.model_path,
                'render_mode': self.render_mode,
                'launched': True,
                'use_legacy': True,
                'nq': self._model.nq,  # Number of generalized coordinates
                'nv': self._model.nv,  # Number of generalized velocities
                'nu': self._model.nu,  # Number of actuators
                'nbody': self._model.nbody,  # Number of bodies
                'timestep': self._model.opt.timestep,
                'gravity': self._model.opt.gravity.tolist()
            }
        else:
            return {
                'model_path': self.model_path,
                'render_mode': self.render_mode,
                'launched': True,
                'use_legacy': False,
                'nq': self._model.nq,
                'nv': self._model.nv,
                'nu': self._model.nu,
                'nbody': self._model.nbody,
                'timestep': self._model.opt.timestep,
                'gravity': self._model.opt.gravity.tolist()
            }
    
    @property
    def is_launched(self) -> bool:
        """Check if the handler has been launched."""
        return self._launched