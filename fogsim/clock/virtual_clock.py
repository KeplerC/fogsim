"""
Virtual Clock - Mode 1 (Virtual Timeline)

Decouples simulation from wallclock time for scalability and reproducibility.
"""

import logging

logger = logging.getLogger(__name__)


class VirtualClock:
    """
    Virtual time clock for FogSim Mode 1.
    
    Provides deterministic, scalable timing by virtualizing the timeline.
    Time advances only when explicitly stepped, enabling high frame rates.
    """
    
    def __init__(self, timestep: float = 0.1):
        """
        Initialize virtual clock.
        
        Args:
            timestep: Time increment per step in seconds
        """
        self.timestep = timestep
        self.time = 0.0
        logger.info(f"VirtualClock initialized with timestep={timestep}")
    
    def advance(self) -> float:
        """
        Advance virtual time by one timestep.
        
        Returns:
            Current virtual time after advancement
        """
        self.time += self.timestep
        return self.time
    
    def now(self) -> float:
        """
        Get current virtual time.
        
        Returns:
            Current virtual time in seconds
        """
        return self.time
    
    def reset(self) -> None:
        """Reset virtual time to zero."""
        self.time = 0.0
        logger.debug("VirtualClock reset to 0.0")
    
    def sleep_until(self, target_time: float) -> None:
        """
        No-op for virtual clock - time is not tied to wallclock.
        
        Args:
            target_time: Target time to sleep until (ignored)
        """
        pass  # Virtual time doesn't sleep