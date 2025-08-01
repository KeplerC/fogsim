"""
Real Network Transport - Mode 3 (Real Network)

Handles actual network communication using an echo server for measuring real round-trip times.
No tc dependency - uses pure network communication.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .echo_server import EchoClient, start_echo_server

logger = logging.getLogger(__name__)


@dataclass 
class NetworkConfig:
    """Network configuration parameters."""
    server_host: str = "127.0.0.1"
    server_port: int = 0  # 0 means auto-select available port
    auto_start_server: bool = True


class RealNetworkTransport:
    """
    Real network transport for FogSim Mode 3.
    
    Uses actual network communication with an echo server to measure real latencies.
    """
    
    def __init__(self, config: Optional[NetworkConfig] = None):
        # Handle both types of NetworkConfig
        if config and hasattr(config, 'server_host'):
            # It's the real network config
            self.config = config
        else:
            # It's the general NetworkConfig, create a real network config
            self.config = NetworkConfig()
        
        self.client = EchoClient(self.config.server_host, self.config.server_port)
        self.server = None
        self.message_count = 0
        self.pending_messages = []
        
        # Auto-start echo server if requested
        if self.config.auto_start_server:
            try:
                self.server = start_echo_server(self.config.server_host, self.config.server_port)
                time.sleep(0.1)  # Let server start
                # Update config with actual port if auto-selected
                self.config.server_port = self.server.port
                self.client = EchoClient(self.config.server_host, self.config.server_port)
                logger.info(f"Started echo server on {self.config.server_host}:{self.config.server_port}")
            except Exception as e:
                logger.warning(f"Failed to start echo server: {e}")
        
        logger.info(f"RealNetworkTransport initialized")
    
    def send_message(self, message: Dict[str, Any]) -> str:
        """
        Send message through real network and measure round-trip time.
        
        Args:
            message: Message to send
            
        Returns:
            Message ID
        """
        msg_id = f"msg_{self.message_count}"
        self.message_count += 1
        
        # Prepare message for sending
        network_message = {
            'id': msg_id,
            'payload': message,
            'step': message.get('step', 0)
        }
        
        # Send to echo server and measure RTT
        response = self.client.send_message(network_message)
        
        if response:
            # Store the response with measured latency
            self.pending_messages.append({
                'id': msg_id,
                'payload': response.get('payload', message),
                'latency': response.get('round_trip_time', 0) * 1000,  # Convert to ms
                'timestamp': time.time()
            })
            
            logger.debug(f"Sent message {msg_id}, RTT: {response.get('round_trip_time', 0)*1000:.1f}ms")
        else:
            # Server not available, simulate local processing
            logger.warning(f"Echo server not available, simulating local message {msg_id}")
            self.pending_messages.append({
                'id': msg_id,
                'payload': message,
                'latency': 0.1,  # 0.1ms for local simulation
                'timestamp': time.time()
            })
        
        return msg_id
    
    def process_messages(self, current_time: float) -> List[Dict[str, Any]]:
        """
        Process received messages.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of messages ready for processing
        """
        # In real network mode, messages are available immediately after RTT
        ready_messages = list(self.pending_messages)
        self.pending_messages.clear()
        
        logger.debug(f"Processed {len(ready_messages)} messages")
        return ready_messages
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network statistics."""
        return {
            'server_host': self.config.server_host,
            'server_port': self.config.server_port,
            'messages_sent': self.message_count,
            'server_running': self.server is not None,
            'client_connected': self.client.connected
        }
    
    def reset(self) -> None:
        """Reset transport state."""
        self.pending_messages.clear()
        self.message_count = 0
        logger.info("RealNetworkTransport reset")
    
    def close(self) -> None:
        """Clean up resources."""
        self.client.close()
        
        if self.server:
            self.server.stop()
            self.server = None
        
        logger.info("RealNetworkTransport closed")