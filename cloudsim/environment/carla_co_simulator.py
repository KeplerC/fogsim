import numpy as np
from typing import Dict, Tuple, Any
from ..base import BaseCoSimulator

class CarlaCoSimulator(BaseCoSimulator):
    """Co-simulator implementation for Carla environments."""
    
    def __init__(self, network_simulator: Any, carla_env: Any, timestep: float = 0.1):
        """
        Initialize the Carla co-simulator.
        
        Args:
            network_simulator: Instance of network simulator (e.g., ns3)
            carla_env: Carla environment instance
            timestep: Simulation timestep in seconds (default: 0.1)
        """
        super().__init__(network_simulator, carla_env, timestep)
        self.current_observation = None
        self.vehicle_sensors = {}  # Dictionary to store vehicle sensor data
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Perform one step of co-simulation.
        
        Args:
            action: Action to be taken in the robotics simulator
            
        Returns:
            Tuple containing observation, reward, done, and info
        """
        # Process any network messages that should be visible now
        self._process_network_messages()
        
        # Step the robotics simulator
        observation, reward, done, info = self.robotics_simulator.step(action)
        self.current_observation = observation
        
        # Send vehicle sensor data through network simulator
        self._send_vehicle_data()
        
        # Advance simulation time
        self._advance_time()
            
        return observation, reward, done, info
    
    def reset(self) -> np.ndarray:
        """Reset both simulators to initial state."""
        observation = self.robotics_simulator.reset()
        self.current_observation = observation
        self.network_simulator.reset()
        self.scheduled_messages.clear()
        self.vehicle_sensors.clear()
        self.current_time = 0.0
        return observation
    
    def render(self, mode: str = 'human') -> None:
        """Render the current state of the simulation."""
        self.robotics_simulator.render(mode=mode)
    
    def _handle_message(self, message: Any) -> None:
        """
        Handle a received message from the network simulator.
        
        Args:
            message: The received message
        """
        if isinstance(message, dict):
            if 'sensor_data' in message:
                # Update vehicle sensor data
                vehicle_id = message.get('vehicle_id')
                if vehicle_id:
                    self.vehicle_sensors[vehicle_id] = message['sensor_data']
            elif 'observation' in message:
                self.current_observation = message['observation']
    
    def _send_vehicle_data(self) -> None:
        """Send vehicle sensor data through the network simulator."""
        # Get all vehicles in the simulation
        vehicles = self.robotics_simulator.get_vehicles()
        
        for vehicle in vehicles:
            # Get sensor data for the vehicle
            sensor_data = self.robotics_simulator.get_vehicle_sensor_data(vehicle)
            
            # Get network latency from ns3
            latency = self.network_simulator.get_latency()
            
            # Schedule the message to be processed after the latency
            message = {
                'vehicle_id': vehicle.id,
                'sensor_data': sensor_data,
                'timestamp': self.get_current_time()
            }
            self._schedule_message(message, latency)
    
    def get_vehicle_sensor_data(self, vehicle_id: str) -> Dict:
        """
        Get the latest sensor data for a specific vehicle.
        
        Args:
            vehicle_id: ID of the vehicle
            
        Returns:
            Dictionary containing sensor data
        """
        return self.vehicle_sensors.get(vehicle_id, {}) 