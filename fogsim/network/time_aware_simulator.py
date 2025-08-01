"""
Time-aware network simulator wrapper that integrates with the unified time backend.
"""

import simpy
import logging
from typing import Any, List, Optional

from .nspy_simulator import NSPyNetworkSimulator, PacketTracker
from ..time_backend import UnifiedTimeManager, SimulationMode, TimeSubscriber
from ..message_passing import MessageBus


logger = logging.getLogger(__name__)


class TimeAwareNetworkSimulator(NSPyNetworkSimulator, TimeSubscriber):
    """
    Network simulator that synchronizes with the unified time backend.
    
    This wrapper ensures that the network simulator's internal time
    stays synchronized with the global simulation time.
    """
    
    def __init__(self, time_manager: UnifiedTimeManager, 
                 source_rate: float = 4600.0, 
                 weights: List[int] = [1, 1], 
                 debug: bool = False):
        """
        Initialize time-aware network simulator.
        
        Args:
            time_manager: Unified time manager instance
            source_rate: Rate in bytes per second
            weights: Weights for different flows
            debug: Enable debug output
        """
        self.time_manager = time_manager
        self.mode = time_manager.mode
        
        # For virtual mode, create a custom SimPy environment
        if self.mode == SimulationMode.VIRTUAL:
            # Virtual mode: SimPy env time directly controlled
            env = VirtualTimeEnvironment(time_manager)
        else:
            # Other modes: standard SimPy environment
            env = simpy.Environment()
        
        # Initialize parent with custom environment
        super().__init__(source_rate, weights, debug, env=env)
        
        # Register with time manager
        self.time_manager.register_subscriber(self)
        
        # Track last sync time
        self._last_sync_time = 0.0
        
        logger.info(f"TimeAwareNetworkSimulator initialized in {self.mode.value} mode")
    
    def sync_to_time(self, time: float) -> None:
        """
        TimeSubscriber interface - synchronize network simulator to global time.
        
        Args:
            time: Global simulation time to sync to
        """
        if self.mode == SimulationMode.VIRTUAL:
            # In virtual mode, the environment is already synced
            # Just process arrivals
            self._process_arrivals(time)
        else:
            # In other modes, run the simulator forward
            if time > self._last_sync_time:
                self.run_until(time)
                self._last_sync_time = time
    
    def calculate_delay(self, sender_id: str, receiver_id: str, 
                       message_size: int) -> float:
        """
        Calculate network delay for a message.
        
        Args:
            sender_id: ID of sender
            receiver_id: ID of receiver
            message_size: Size of message in bytes
            
        Returns:
            Calculated delay in seconds
        """
        # Basic calculation based on virtual clock scheduler
        # This is a simplified model - could be extended
        base_delay = 0.001  # 1ms base propagation delay
        
        # Bandwidth-based delay (assuming source_rate is in bytes/sec)
        bandwidth_delay = message_size / self.source_rate
        
        # Add queueing delay estimate
        # In real implementation, this would consider current queue state
        queueing_delay = 0.001  # 1ms estimated queueing
        
        total_delay = base_delay + bandwidth_delay + queueing_delay
        
        logger.debug(f"Calculated delay: {total_delay}s for {message_size} bytes")
        return total_delay
    
    def configure_link(self, config: dict) -> None:
        """
        Configure network link parameters.
        
        Args:
            config: Dictionary with link parameters
                - delay: Base delay in seconds
                - bandwidth: Bandwidth in bits/sec
                - loss: Loss probability (0-1)
                - jitter: Jitter in seconds
        """
        if 'bandwidth' in config and config['bandwidth']:
            # Convert from bits/sec to bytes/sec
            self.source_rate = config['bandwidth'] / 8.0
            
            # Recreate virtual clock server with new rate
            self.vc_server = VirtualClockServer(
                self.env, 
                self.source_rate, 
                self.flow_weights, 
                debug=self.debug
            )
            self.vc_server.out = self.sink
        
        # Store other parameters for delay calculation
        self.link_config = config
        
        logger.info(f"Configured link with: {config}")
    
    def get_stats(self) -> dict:
        """Get network simulator statistics."""
        stats = {
            'mode': self.mode.value,
            'current_time': self.env.now,
            'source_rate': self.source_rate,
            'pending_packets': len(self.packet_tracker.pending_packets),
            'total_arrivals': sum(len(arrivals) for arrivals in self.sink.arrivals.values())
        }
        
        # Add per-flow statistics
        flow_stats = {}
        for flow_id, arrivals in self.sink.arrivals.items():
            flow_stats[f'flow_{flow_id}_arrivals'] = len(arrivals)
        
        stats.update(flow_stats)
        return stats
    
    def close(self) -> None:
        """Clean up resources."""
        # Unregister from time manager
        self.time_manager.unregister_subscriber(self)
        super().close()


class VirtualTimeEnvironment(simpy.Environment):
    """
    Custom SimPy environment that uses virtual time backend.
    
    This environment synchronizes with the UnifiedTimeManager
    instead of using its own internal clock.
    """
    
    def __init__(self, time_manager: UnifiedTimeManager):
        super().__init__()
        self.time_manager = time_manager
        self._now = time_manager.now()
    
    @property
    def now(self) -> float:
        """Get current time from time manager."""
        return self.time_manager.now()
    
    def run(self, until: Optional[float] = None) -> None:
        """
        Run simulation until specified time.
        
        In virtual mode, this doesn't actually advance time
        but processes events scheduled up to the target time.
        """
        if until is None:
            return
        
        # Process all events up to target time
        while self._queue and self._queue[0][0] <= until:
            # Get next event
            event_time, _, event_id, event = self._queue.pop(0)
            
            # Process the event
            self._now = event_time
            event._ok = True
            event._value = None
            
            # Trigger all callbacks
            for callback in event._callbacks:
                callback(event)
        
        # Update internal time
        self._now = until


def create_network_simulator(time_manager: UnifiedTimeManager,
                           network_config: Optional[Any] = None) -> TimeAwareNetworkSimulator:
    """
    Factory function to create appropriate network simulator based on mode.
    
    Args:
        time_manager: Unified time manager
        network_config: Network configuration
        
    Returns:
        Configured network simulator
    """
    # Extract parameters from config if provided
    if network_config:
        source_rate = getattr(network_config, 'source_rate', 4600.0)
        weights = getattr(network_config, 'flow_weights', [1, 1])
        debug = getattr(network_config, 'debug', False)
    else:
        source_rate = 4600.0
        weights = [1, 1]
        debug = False
    
    # Create time-aware simulator
    simulator = TimeAwareNetworkSimulator(
        time_manager=time_manager,
        source_rate=source_rate,
        weights=weights,
        debug=debug
    )
    
    return simulator