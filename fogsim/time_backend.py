"""
Time Backend System for FogSim

This module provides the core time abstraction layer that enables FogSim to operate
in three different modes:
1. Virtual Time (FogSIM) - Pure virtual timeline for high performance
2. Simulated Network - Wallclock with simulated network delays
3. Real Network - Wallclock with real network using Linux tc
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple
import heapq
import time
import asyncio
import logging
from threading import Lock


logger = logging.getLogger(__name__)


class SimulationMode(Enum):
    """Three operational modes for FogSim"""
    VIRTUAL = "virtual"  # Mode 1: Pure virtual time
    SIMULATED_NET = "simulated_net"  # Mode 2: Wallclock + simulated network
    REAL_NET = "real_net"  # Mode 3: Wallclock + real network


@dataclass
class ScheduledEvent:
    """Event scheduled for future execution"""
    time: float
    callback: Callable
    priority: int = 0
    
    def __lt__(self, other):
        # Earlier time or higher priority comes first
        if self.time != other.time:
            return self.time < other.time
        return self.priority > other.priority


class TimeBackend(ABC):
    """Abstract interface for different time management strategies"""
    
    @abstractmethod
    def now(self) -> float:
        """Get current time in seconds"""
        pass
    
    @abstractmethod
    def sleep(self, duration: float) -> None:
        """Sleep for specified duration"""
        pass
    
    @abstractmethod
    def schedule_event(self, time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule an event to occur at a specific time"""
        pass
    
    @abstractmethod
    def run_until(self, target_time: float) -> None:
        """Run simulation until target time is reached"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset time to initial state"""
        pass


class VirtualTimeBackend(TimeBackend):
    """
    Mode 1: Pure virtual time backend
    - No wallclock coupling
    - Deterministic event scheduling
    - Time advances only on explicit calls
    - Highest performance and reproducibility
    """
    
    def __init__(self, initial_time: float = 0.0):
        self._current_time = initial_time
        self._initial_time = initial_time
        self._event_queue: List[ScheduledEvent] = []
        self._lock = Lock()
        
    def now(self) -> float:
        """Get current virtual time"""
        with self._lock:
            return self._current_time
    
    def sleep(self, duration: float) -> None:
        """Advance virtual time by duration (no actual sleep)"""
        with self._lock:
            self._current_time += duration
            self._process_events()
    
    def schedule_event(self, time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule event in virtual time"""
        with self._lock:
            if time < self._current_time:
                logger.warning(f"Scheduling event in the past: {time} < {self._current_time}")
            heapq.heappush(self._event_queue, ScheduledEvent(time, callback, priority))
    
    def run_until(self, target_time: float) -> None:
        """Process all events until target time"""
        with self._lock:
            while self._event_queue and self._event_queue[0].time <= target_time:
                event = heapq.heappop(self._event_queue)
                self._current_time = event.time
                try:
                    event.callback()
                except Exception as e:
                    logger.error(f"Error in scheduled event: {e}")
            
            # Advance to target time if no more events
            if target_time > self._current_time:
                self._current_time = target_time
    
    def reset(self) -> None:
        """Reset to initial time"""
        with self._lock:
            self._current_time = self._initial_time
            self._event_queue.clear()
    
    def _process_events(self) -> None:
        """Process events up to current time"""
        while self._event_queue and self._event_queue[0].time <= self._current_time:
            event = heapq.heappop(self._event_queue)
            try:
                event.callback()
            except Exception as e:
                logger.error(f"Error in scheduled event: {e}")


class SimulatedNetworkTimeBackend(TimeBackend):
    """
    Mode 2: Wallclock with simulated network
    - Synchronized to wallclock time
    - Network delays from ns.py simulator
    - Frame rate limited by wallclock
    """
    
    def __init__(self):
        self._start_wallclock = time.time()
        self._start_sim_time = 0.0
        self._event_queue: List[ScheduledEvent] = []
        self._lock = Lock()
        self._time_scale = 1.0  # Real-time by default
        
    def now(self) -> float:
        """Get current time synchronized with wallclock"""
        elapsed_wallclock = time.time() - self._start_wallclock
        return self._start_sim_time + (elapsed_wallclock * self._time_scale)
    
    def sleep(self, duration: float) -> None:
        """Sleep for actual wallclock duration"""
        time.sleep(duration / self._time_scale)
        self._process_events()
    
    def schedule_event(self, sim_time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule event based on simulation time"""
        with self._lock:
            heapq.heappush(self._event_queue, ScheduledEvent(sim_time, callback, priority))
    
    def run_until(self, target_time: float) -> None:
        """Run until target simulation time, respecting wallclock"""
        while self.now() < target_time:
            # Process any pending events
            self._process_events()
            
            # Calculate remaining time
            remaining = target_time - self.now()
            if remaining > 0:
                # Sleep for a small interval or until next event
                sleep_duration = min(0.001, remaining)  # 1ms resolution
                time.sleep(sleep_duration / self._time_scale)
    
    def reset(self) -> None:
        """Reset time tracking"""
        with self._lock:
            self._start_wallclock = time.time()
            self._start_sim_time = 0.0
            self._event_queue.clear()
    
    def set_time_scale(self, scale: float) -> None:
        """Set time scaling factor (1.0 = real-time, 2.0 = 2x speed)"""
        current_time = self.now()
        self._time_scale = scale
        self._start_wallclock = time.time()
        self._start_sim_time = current_time
    
    def _process_events(self) -> None:
        """Process events up to current time"""
        current = self.now()
        with self._lock:
            while self._event_queue and self._event_queue[0].time <= current:
                event = heapq.heappop(self._event_queue)
                try:
                    event.callback()
                except Exception as e:
                    logger.error(f"Error in scheduled event: {e}")


class RealNetworkTimeBackend(TimeBackend):
    """
    Mode 3: Wallclock with real network
    - Synchronized to wallclock time
    - Real network communication
    - Linux tc for network control
    """
    
    def __init__(self):
        self._start_wallclock = time.time()
        self._event_loop = None
        self._scheduled_tasks: Dict[asyncio.Task, float] = {}
        
    def now(self) -> float:
        """Get current wallclock time since start"""
        return time.time() - self._start_wallclock
    
    def sleep(self, duration: float) -> None:
        """Sleep for actual wallclock duration"""
        time.sleep(duration)
    
    def schedule_event(self, time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule event using asyncio for real-time execution"""
        delay = time - self.now()
        if delay < 0:
            logger.warning(f"Scheduling event in the past: {delay}s ago")
            delay = 0
        
        # Use threading timer for simplicity in sync context
        import threading
        timer = threading.Timer(delay, callback)
        timer.start()
    
    def run_until(self, target_time: float) -> None:
        """Wait until target wallclock time"""
        remaining = target_time - self.now()
        if remaining > 0:
            time.sleep(remaining)
    
    def reset(self) -> None:
        """Reset time tracking"""
        self._start_wallclock = time.time()
        # Cancel any pending timers if needed


class UnifiedTimeManager:
    """
    Unified time management system that coordinates time across all components
    """
    
    def __init__(self, mode: SimulationMode, timestep: float = 0.1):
        self.mode = mode
        self.timestep = timestep
        self.subscribers: List['TimeSubscriber'] = []
        
        # Create appropriate backend
        if mode == SimulationMode.VIRTUAL:
            self.backend = VirtualTimeBackend(initial_time=timestep)
        elif mode == SimulationMode.SIMULATED_NET:
            self.backend = SimulatedNetworkTimeBackend()
        else:  # REAL_NET
            self.backend = RealNetworkTimeBackend()
        
        self._lock = Lock()
        
    def register_subscriber(self, subscriber: 'TimeSubscriber') -> None:
        """Register a component that needs time updates"""
        with self._lock:
            self.subscribers.append(subscriber)
    
    def unregister_subscriber(self, subscriber: 'TimeSubscriber') -> None:
        """Unregister a time subscriber"""
        with self._lock:
            self.subscribers.remove(subscriber)
    
    def synchronize(self) -> None:
        """Synchronize all components to current time"""
        current = self.backend.now()
        with self._lock:
            for subscriber in self.subscribers:
                subscriber.sync_to_time(current)
    
    def advance_time(self) -> None:
        """Advance time by one timestep"""
        target_time = self.backend.now() + self.timestep
        self.backend.run_until(target_time)
        self.synchronize()
    
    def now(self) -> float:
        """Get current time from backend"""
        return self.backend.now()
    
    def schedule_event(self, time: float, callback: Callable, priority: int = 0) -> None:
        """Schedule an event through the backend"""
        self.backend.schedule_event(time, callback, priority)
    
    def run_until(self, time: float) -> None:
        """Run until specified time"""
        self.backend.run_until(time)
        self.synchronize()
    
    def reset(self) -> None:
        """Reset time system"""
        self.backend.reset()
        self.synchronize()


class TimeSubscriber(ABC):
    """Interface for components that need time synchronization"""
    
    @abstractmethod
    def sync_to_time(self, time: float) -> None:
        """Synchronize component to specified time"""
        pass