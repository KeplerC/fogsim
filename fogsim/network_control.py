"""
Network Control Layer for FogSim

This module provides network control capabilities for different simulation modes:
- Simulated network control (via ns.py)
- Real network control (via Linux tc)
"""

import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from .time_backend import SimulationMode


logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Network configuration parameters"""
    delay: float = 0.0  # Delay in milliseconds
    jitter: float = 0.0  # Jitter in milliseconds
    bandwidth: Optional[float] = None  # Bandwidth in Mbps
    loss: float = 0.0  # Packet loss percentage
    reorder: float = 0.0  # Packet reorder percentage
    duplicate: float = 0.0  # Packet duplication percentage
    corrupt: float = 0.0  # Packet corruption percentage


class NetworkController(ABC):
    """Abstract interface for network control"""
    
    @abstractmethod
    def configure(self, config: NetworkConfig) -> None:
        """Apply network configuration"""
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset network to default state"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        pass


class TCController(NetworkController):
    """
    Linux Traffic Control (tc) wrapper for real network control
    Used in Mode 3 (REAL_NET)
    """
    
    def __init__(self, interface: str = "lo"):
        self.interface = interface
        self._original_config_saved = False
        self._check_tc_available()
    
    def _check_tc_available(self) -> None:
        """Check if tc command is available"""
        try:
            subprocess.run(["tc", "-Version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("tc (traffic control) command not available. "
                             "Please install iproute2 package.")
    
    def _run_tc_command(self, args: list) -> subprocess.CompletedProcess:
        """Run tc command with error handling"""
        cmd = ["sudo", "tc"] + args
        logger.debug(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, 
                                  text=True, check=True)
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"tc command failed: {e.stderr}")
            raise
    
    def configure(self, config: NetworkConfig) -> None:
        """Apply network configuration using tc"""
        # Clear existing configuration
        self.reset()
        
        # Add root qdisc
        self._run_tc_command([
            "qdisc", "add", "dev", self.interface, 
            "root", "handle", "1:", "htb"
        ])
        
        # Add class with bandwidth limit if specified
        if config.bandwidth:
            rate = f"{config.bandwidth}mbit"
            self._run_tc_command([
                "class", "add", "dev", self.interface,
                "parent", "1:", "classid", "1:1", "htb",
                "rate", rate
            ])
            parent = "1:1"
        else:
            parent = "1:"
        
        # Build netem parameters
        netem_args = [
            "qdisc", "add", "dev", self.interface,
            "parent", parent, "handle", "10:", "netem"
        ]
        
        # Add delay
        if config.delay > 0:
            netem_args.extend(["delay", f"{config.delay}ms"])
            if config.jitter > 0:
                netem_args.extend([f"{config.jitter}ms"])
        
        # Add packet loss
        if config.loss > 0:
            netem_args.extend(["loss", f"{config.loss}%"])
        
        # Add reordering
        if config.reorder > 0:
            netem_args.extend(["reorder", f"{config.reorder}%"])
        
        # Add duplication
        if config.duplicate > 0:
            netem_args.extend(["duplicate", f"{config.duplicate}%"])
        
        # Add corruption
        if config.corrupt > 0:
            netem_args.extend(["corrupt", f"{config.corrupt}%"])
        
        # Apply netem configuration
        if len(netem_args) > 7:  # Has parameters beyond basic command
            self._run_tc_command(netem_args)
        
        logger.info(f"Applied network config to {self.interface}: {config}")
    
    def reset(self) -> None:
        """Reset network to default state"""
        try:
            # Delete existing qdisc (this removes all child qdiscs too)
            self._run_tc_command([
                "qdisc", "del", "dev", self.interface, "root"
            ])
            logger.info(f"Reset network configuration on {self.interface}")
        except subprocess.CalledProcessError:
            # No existing qdisc, ignore error
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics from tc"""
        try:
            # Get qdisc statistics
            result = self._run_tc_command([
                "qdisc", "show", "dev", self.interface
            ])
            
            stats = {
                "interface": self.interface,
                "qdisc_info": result.stdout
            }
            
            # Get detailed statistics
            result = self._run_tc_command([
                "-s", "qdisc", "show", "dev", self.interface
            ])
            stats["detailed_stats"] = result.stdout
            
            return stats
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}


class SimulatedNetworkController(NetworkController):
    """
    Network control for simulated network (ns.py)
    Used in Mode 2 (SIMULATED_NET)
    """
    
    def __init__(self, network_simulator):
        self.network_simulator = network_simulator
        self.current_config = NetworkConfig()
    
    def configure(self, config: NetworkConfig) -> None:
        """Apply network configuration to simulator"""
        # Update network simulator parameters
        if hasattr(self.network_simulator, 'configure_link'):
            link_config = {
                'delay': config.delay / 1000.0,  # Convert to seconds
                'bandwidth': config.bandwidth * 1e6 if config.bandwidth else None,
                'loss': config.loss / 100.0,  # Convert to probability
                'jitter': config.jitter / 1000.0
            }
            self.network_simulator.configure_link(link_config)
        
        self.current_config = config
        logger.info(f"Applied simulated network config: {config}")
    
    def reset(self) -> None:
        """Reset network simulator to default"""
        self.configure(NetworkConfig())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from network simulator"""
        if hasattr(self.network_simulator, 'get_stats'):
            return self.network_simulator.get_stats()
        return {
            "current_config": self.current_config,
            "stats": "Not available"
        }


class NetworkControlManager:
    """
    Manages network control based on simulation mode
    """
    
    def __init__(self, mode: SimulationMode, interface: str = "lo", 
                 network_simulator=None):
        self.mode = mode
        
        if mode == SimulationMode.REAL_NET:
            self.controller = TCController(interface)
        elif mode == SimulationMode.SIMULATED_NET:
            if network_simulator is None:
                raise ValueError("Network simulator required for SIMULATED_NET mode")
            self.controller = SimulatedNetworkController(network_simulator)
        else:  # VIRTUAL mode
            # No network control needed for pure virtual time
            self.controller = None
    
    def configure(self, config: NetworkConfig) -> None:
        """Apply network configuration if applicable"""
        if self.controller:
            self.controller.configure(config)
    
    def reset(self) -> None:
        """Reset network configuration"""
        if self.controller:
            self.controller.reset()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        if self.controller:
            return self.controller.get_stats()
        return {"mode": "VIRTUAL", "stats": "N/A"}
    
    @staticmethod
    def create_real_network_configs() -> Dict[str, NetworkConfig]:
        """Predefined network configurations for real network testing"""
        return {
            "low_latency": NetworkConfig(
                delay=1.0,  # 1ms
                jitter=0.1,
                bandwidth=1000,  # 1Gbps
                loss=0.0
            ),
            "edge_cloud": NetworkConfig(
                delay=10.0,  # 10ms
                jitter=2.0,
                bandwidth=100,  # 100Mbps
                loss=0.1
            ),
            "wan": NetworkConfig(
                delay=50.0,  # 50ms
                jitter=10.0,
                bandwidth=50,  # 50Mbps
                loss=0.5
            ),
            "satellite": NetworkConfig(
                delay=600.0,  # 600ms
                jitter=50.0,
                bandwidth=10,  # 10Mbps
                loss=1.0
            ),
            "congested": NetworkConfig(
                delay=100.0,
                jitter=50.0,
                bandwidth=5,  # 5Mbps
                loss=2.0,
                reorder=1.0
            )
        }