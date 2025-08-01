"""
Real Clock - Mode 2 & 3 (Real Time Synchronization)

Synchronizes simulation with wallclock time for network simulation accuracy.
"""

import time
import logging

logger = logging.getLogger(__name__)


class RealClock:
    """
    Real time clock for FogSim Mode 2 and 3.
    
    Synchronizes simulation steps with wallclock time to enable:
    - Accurate network delay simulation (Mode 2)
    - Real network communication (Mode 3)
    """
    
    def __init__(self, timestep: float = 0.1):
        """
        Initialize real clock.
        
        Args:
            timestep: Target time between steps in seconds
        """
        self.timestep = timestep
        self.start_time = time.time()
        self.step_count = 0
        logger.info(f"RealClock initialized with timestep={timestep}")
    
    def advance(self) -> float:
        """
        Advance to next timestep, sleeping if necessary to maintain real-time sync.
        
        Returns:
            Current simulation time after advancement
        """
        self.step_count += 1
        target_time = self.start_time + (self.step_count * self.timestep)
        current_time = time.time()
        
        # Sleep if we're ahead of schedule
        if current_time < target_time:
            sleep_duration = target_time - current_time
            time.sleep(sleep_duration)
        
        return self.now()
    
    def now(self) -> float:
        """
        Get current simulation time.
        
        Returns:
            Current simulation time in seconds (relative to start)
        """
        return self.step_count * self.timestep
    
    def reset(self) -> None:
        """Reset clock to start new simulation."""
        self.start_time = time.time() 
        self.step_count = 0
        logger.debug("RealClock reset")
    
    def sleep_until(self, target_time: float) -> None:
        """
        Sleep until specific simulation time.
        
        Args:
            target_time: Target simulation time to sleep until
        """
        wallclock_target = self.start_time + target_time
        current_time = time.time()
        
        if current_time < wallclock_target:
            sleep_duration = wallclock_target - current_time
            time.sleep(sleep_duration)