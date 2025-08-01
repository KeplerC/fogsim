"""
Real Network Mode Support for FogSim

This module provides flexible real network configuration options:
1. Automatic tc configuration (requires sudo)
2. Manual tc configuration (user manages tc)
3. Latency measurement from actual network endpoints
"""

import subprocess
import time
import re
import logging
import statistics
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from .network_control import NetworkConfig, NetworkController


logger = logging.getLogger(__name__)


@dataclass
class NetworkMeasurement:
    """Results from network measurement"""
    target: str
    avg_latency: float  # ms
    min_latency: float  # ms
    max_latency: float  # ms
    jitter: float  # ms
    packet_loss: float  # percentage
    bandwidth: Optional[float] = None  # Mbps


class LatencyMeasurer:
    """Measure real network latency to remote hosts"""
    
    def __init__(self):
        self.measurements: Dict[str, NetworkMeasurement] = {}
    
    def measure_latency(self, target: str, count: int = 10) -> NetworkMeasurement:
        """
        Measure network latency to a target host using ping.
        
        Args:
            target: Hostname or IP address to ping
            count: Number of ping packets to send
            
        Returns:
            NetworkMeasurement with latency statistics
        """
        logger.info(f"Measuring latency to {target} with {count} pings...")
        
        try:
            # Run ping command
            cmd = ["ping", "-c", str(count), "-i", "0.2", target]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse ping output
            latencies = []
            for line in result.stdout.splitlines():
                # Look for lines with "time=XX.X ms"
                match = re.search(r'time=(\d+\.?\d*)\s*ms', line)
                if match:
                    latencies.append(float(match.group(1)))
            
            if not latencies:
                raise ValueError("No latency measurements found in ping output")
            
            # Calculate statistics
            avg_latency = statistics.mean(latencies)
            min_latency = min(latencies)
            max_latency = max(latencies)
            jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            
            # Parse packet loss from summary line
            packet_loss = 0.0
            loss_match = re.search(r'(\d+(?:\.\d+)?)%\s+packet loss', result.stdout)
            if loss_match:
                packet_loss = float(loss_match.group(1))
            
            measurement = NetworkMeasurement(
                target=target,
                avg_latency=avg_latency,
                min_latency=min_latency,
                max_latency=max_latency,
                jitter=jitter,
                packet_loss=packet_loss
            )
            
            self.measurements[target] = measurement
            logger.info(f"Latency to {target}: {avg_latency:.1f}ms (±{jitter:.1f}ms), "
                       f"loss: {packet_loss:.1f}%")
            
            return measurement
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Ping failed: {e}")
            raise RuntimeError(f"Failed to measure latency to {target}: {e}")
        except Exception as e:
            logger.error(f"Error measuring latency: {e}")
            raise
    
    def measure_bandwidth(self, target: str, port: int = 5201, 
                         duration: int = 5) -> Optional[float]:
        """
        Measure bandwidth using iperf3 (if available).
        
        Args:
            target: iperf3 server hostname/IP
            port: iperf3 server port
            duration: Test duration in seconds
            
        Returns:
            Bandwidth in Mbps or None if iperf3 not available
        """
        try:
            cmd = ["iperf3", "-c", target, "-p", str(port), "-t", str(duration), "-J"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Parse JSON output
                import json
                data = json.loads(result.stdout)
                # Get average bandwidth in bits/sec, convert to Mbps
                bandwidth_bps = data['end']['sum_received']['bits_per_second']
                bandwidth_mbps = bandwidth_bps / 1e6
                
                logger.info(f"Bandwidth to {target}: {bandwidth_mbps:.1f} Mbps")
                return bandwidth_mbps
            else:
                logger.warning(f"iperf3 test failed: {result.stderr}")
                return None
                
        except FileNotFoundError:
            logger.info("iperf3 not available for bandwidth measurement")
            return None
        except Exception as e:
            logger.warning(f"Error measuring bandwidth: {e}")
            return None


class ManualTCController(NetworkController):
    """
    Manual TC mode - user configures tc themselves.
    This controller only provides configuration guidance.
    """
    
    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.config_applied = False
    
    def configure(self, config: NetworkConfig) -> None:
        """Print tc commands for manual configuration"""
        print("\n" + "="*70)
        print("MANUAL TC CONFIGURATION")
        print("="*70)
        print(f"To simulate the requested network conditions on {self.interface}:")
        print("\n# First, clear any existing configuration:")
        print(f"sudo tc qdisc del dev {self.interface} root 2>/dev/null")
        
        print("\n# Add root qdisc:")
        print(f"sudo tc qdisc add dev {self.interface} root handle 1: htb default 1")
        
        if config.bandwidth:
            print(f"\n# Add bandwidth limit ({config.bandwidth} Mbps):")
            print(f"sudo tc class add dev {self.interface} parent 1: classid 1:1 htb "
                  f"rate {config.bandwidth}mbit")
            parent = "1:1"
        else:
            parent = "1:"
        
        # Build netem command
        netem_cmd = f"sudo tc qdisc add dev {self.interface} parent {parent} handle 10: netem"
        
        if config.delay > 0:
            netem_cmd += f" delay {config.delay:.1f}ms"
            if config.jitter > 0:
                netem_cmd += f" {config.jitter:.1f}ms"
        
        if config.loss > 0:
            netem_cmd += f" loss {config.loss:.1f}%"
        
        if config.reorder > 0:
            netem_cmd += f" reorder {config.reorder:.1f}%"
        
        if config.duplicate > 0:
            netem_cmd += f" duplicate {config.duplicate:.1f}%"
        
        if config.corrupt > 0:
            netem_cmd += f" corrupt {config.corrupt:.1f}%"
        
        print(f"\n# Add network emulation:")
        print(netem_cmd)
        
        print("\n# To remove these settings later:")
        print(f"sudo tc qdisc del dev {self.interface} root")
        
        print("\n# To verify configuration:")
        print(f"tc qdisc show dev {self.interface}")
        print("="*70 + "\n")
        
        self.config_applied = True
        logger.info(f"Manual TC configuration displayed for {self.interface}")
    
    def reset(self) -> None:
        """Show reset commands"""
        if self.config_applied:
            print(f"\nTo reset network configuration:")
            print(f"sudo tc qdisc del dev {self.interface} root")
            self.config_applied = False
    
    def get_stats(self) -> Dict[str, any]:
        """Get current tc statistics"""
        try:
            result = subprocess.run(
                ["tc", "-s", "qdisc", "show", "dev", self.interface],
                capture_output=True, text=True
            )
            return {
                "interface": self.interface,
                "tc_output": result.stdout,
                "manual_mode": True
            }
        except:
            return {"interface": self.interface, "error": "Could not read tc stats"}


class RealNetworkManager:
    """
    Manages real network configuration with multiple options:
    1. Automatic tc (requires sudo)
    2. Manual tc (user configures)
    3. Latency measurement from actual endpoints
    """
    
    def __init__(self, mode: str = "manual", interface: str = "eth0"):
        """
        Initialize real network manager.
        
        Args:
            mode: "auto" for automatic tc, "manual" for user-managed tc
            interface: Network interface to configure
        """
        self.mode = mode
        self.interface = interface
        self.latency_measurer = LatencyMeasurer()
        
        if mode == "auto":
            from .network_control import TCController
            self.controller = TCController(interface)
        else:
            self.controller = ManualTCController(interface)
    
    def measure_and_configure(self, target: str, scale_factor: float = 1.0) -> NetworkConfig:
        """
        Measure real network characteristics and create configuration.
        
        Args:
            target: Remote host to measure
            scale_factor: Scale measured values (e.g., 2.0 for 2x latency)
            
        Returns:
            NetworkConfig based on measurements
        """
        # Measure real network
        measurement = self.latency_measurer.measure_latency(target)
        
        # Optionally measure bandwidth
        bandwidth = self.latency_measurer.measure_bandwidth(target)
        
        # Create config based on measurements
        config = NetworkConfig(
            delay=measurement.avg_latency * scale_factor,
            jitter=measurement.jitter * scale_factor,
            bandwidth=bandwidth / scale_factor if bandwidth else None,
            loss=measurement.packet_loss * scale_factor
        )
        
        logger.info(f"Created config from {target} measurements (scale={scale_factor}): {config}")
        
        # Apply configuration
        self.controller.configure(config)
        
        return config
    
    def configure_from_profile(self, profile: str) -> NetworkConfig:
        """
        Configure network based on predefined profiles.
        
        Args:
            profile: Profile name (e.g., "lan", "wan", "satellite", "3g", "4g")
            
        Returns:
            Applied NetworkConfig
        """
        profiles = {
            "lan": NetworkConfig(delay=0.5, jitter=0.1, bandwidth=1000, loss=0.0),
            "wifi": NetworkConfig(delay=5.0, jitter=2.0, bandwidth=100, loss=0.1),
            "wan": NetworkConfig(delay=30.0, jitter=5.0, bandwidth=100, loss=0.2),
            "3g": NetworkConfig(delay=150.0, jitter=50.0, bandwidth=2, loss=1.0),
            "4g": NetworkConfig(delay=50.0, jitter=10.0, bandwidth=20, loss=0.5),
            "satellite": NetworkConfig(delay=600.0, jitter=50.0, bandwidth=10, loss=2.0),
            "edge": NetworkConfig(delay=10.0, jitter=2.0, bandwidth=100, loss=0.1),
            "cloud": NetworkConfig(delay=50.0, jitter=10.0, bandwidth=1000, loss=0.1)
        }
        
        if profile not in profiles:
            raise ValueError(f"Unknown profile: {profile}. Available: {list(profiles.keys())}")
        
        config = profiles[profile]
        logger.info(f"Using {profile} profile: {config}")
        
        self.controller.configure(config)
        return config
    
    def get_example_targets(self) -> Dict[str, str]:
        """Get example targets for latency measurement"""
        return {
            "local": "127.0.0.1",
            "gateway": "192.168.1.1",  # Common router IP
            "dns": "8.8.8.8",  # Google DNS
            "cloudflare": "1.1.1.1",  # Cloudflare DNS
            "google": "google.com",
            "aws_us_east": "ec2.us-east-1.amazonaws.com",
            "aws_eu": "ec2.eu-west-1.amazonaws.com",
            "aws_asia": "ec2.ap-southeast-1.amazonaws.com"
        }


def create_real_network_config(target: Optional[str] = None, 
                             profile: Optional[str] = None,
                             mode: str = "manual",
                             interface: str = "eth0") -> Tuple[NetworkConfig, RealNetworkManager]:
    """
    Helper function to create real network configuration.
    
    Args:
        target: Remote host to measure latency from
        profile: Predefined profile name
        mode: "auto" or "manual" tc configuration
        interface: Network interface
        
    Returns:
        Tuple of (NetworkConfig, RealNetworkManager)
    """
    manager = RealNetworkManager(mode=mode, interface=interface)
    
    if target:
        # Measure real network to target
        config = manager.measure_and_configure(target)
    elif profile:
        # Use predefined profile
        config = manager.configure_from_profile(profile)
    else:
        # Default configuration
        config = NetworkConfig(delay=10.0, jitter=2.0, bandwidth=100, loss=0.1)
        manager.controller.configure(config)
    
    return config, manager