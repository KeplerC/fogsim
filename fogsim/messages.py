"""
Simple Message Definitions for FogSim
"""

from dataclasses import dataclass
from typing import Any, Dict
import time


@dataclass
class Message:
    """Simple message container."""
    id: str
    sender: str
    receiver: str
    payload: Any
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class SimulationMessage(Message):
    """Message with simulation-specific fields."""
    simulation_time: float = 0.0
    step: int = 0
    latency: float = 0.0  # Network latency in ms


def create_step_message(step: int, observation: Any, action: Any, 
                       sender: str = "agent", receiver: str = "server") -> SimulationMessage:
    """
    Create a standard simulation step message.
    
    Args:
        step: Current simulation step
        observation: Current observation
        action: Action taken
        sender: Message sender ID
        receiver: Message receiver ID
        
    Returns:
        SimulationMessage
    """
    return SimulationMessage(
        id=f"step_{step}_{int(time.time()*1000)}",
        sender=sender,
        receiver=receiver,
        payload={
            'observation': observation,
            'action': action,
            'step': step
        },
        step=step
    )