"""
FogSim Real Network Client

This client forwards simulation messages through a real network server
to measure actual network latency and behavior.
"""

import asyncio
import json
import time
import socket
import struct
import logging
import uuid
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict
from collections import deque
import threading
import queue
import numpy as np

from .time_backend import TimeSubscriber
from .message_passing import MessageHandler, TimedMessage

logger = logging.getLogger(__name__)


@dataclass
class NetworkStats:
    """Statistics for network performance"""
    messages_sent: int = 0
    messages_received: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    packet_loss: int = 0
    latencies: List[float] = None
    
    def __post_init__(self):
        if self.latencies is None:
            self.latencies = []
    
    def add_latency(self, latency: float):
        """Add a latency measurement"""
        self.latencies.append(latency)
        self.total_latency += latency
        self.min_latency = min(self.min_latency, latency)
        self.max_latency = max(self.max_latency, latency)
        self.messages_received += 1
    
    def get_average_latency(self) -> float:
        """Get average latency in ms"""
        if self.messages_received > 0:
            return self.total_latency / self.messages_received
        return 0.0
    
    def get_packet_loss_rate(self) -> float:
        """Get packet loss rate as percentage"""
        if self.messages_sent > 0:
            return (self.packet_loss / self.messages_sent) * 100
        return 0.0


class RealNetworkClient:
    """
    Client for forwarding messages through real network server
    """
    
    def __init__(self, server_host: str = "127.0.0.1", 
                 server_port: int = 8765,
                 protocol: str = "tcp",
                 timeout: float = 5.0):
        self.server_host = server_host
        self.server_port = server_port
        self.protocol = protocol
        self.timeout = timeout
        
        self.stats = NetworkStats()
        self.pending_messages: Dict[str, float] = {}  # message_id -> send_time
        self.response_queue = queue.Queue()
        
        # TCP connection
        self.tcp_reader: Optional[asyncio.StreamReader] = None
        self.tcp_writer: Optional[asyncio.StreamWriter] = None
        
        # UDP socket
        self.udp_socket: Optional[socket.socket] = None
        
        # Background thread for async operations
        self.running = False
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        
    def start(self):
        """Start the client connection"""
        self.running = True
        
        if self.protocol == "tcp":
            # Start async event loop in background thread
            self.thread = threading.Thread(target=self._run_tcp_client, daemon=True)
            self.thread.start()
            
            # Wait for connection
            time.sleep(1.0)
            
        elif self.protocol == "udp":
            # Create UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.settimeout(0.1)  # Non-blocking with short timeout
            
            # Start receiver thread
            self.thread = threading.Thread(target=self._run_udp_receiver, daemon=True)
            self.thread.start()
    
    def _run_tcp_client(self):
        """Run TCP client in async event loop"""
        self.event_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_until_complete(self._tcp_client_loop())
    
    async def _tcp_client_loop(self):
        """TCP client main loop"""
        try:
            # Connect to server
            self.tcp_reader, self.tcp_writer = await asyncio.open_connection(
                self.server_host, self.server_port
            )
            logger.info(f"Connected to TCP server at {self.server_host}:{self.server_port}")
            
            # Start receiver task
            receiver_task = asyncio.create_task(self._tcp_receiver())
            
            # Keep running until stopped
            while self.running:
                await asyncio.sleep(0.1)
            
            # Cleanup
            receiver_task.cancel()
            self.tcp_writer.close()
            await self.tcp_writer.wait_closed()
            
        except Exception as e:
            logger.error(f"TCP client error: {e}")
    
    async def _tcp_receiver(self):
        """Receive messages from TCP server"""
        try:
            while self.running:
                # Read message length
                length_data = await self.tcp_reader.readexactly(4)
                msg_length = struct.unpack("!I", length_data)[0]
                
                # Read message
                data = await self.tcp_reader.readexactly(msg_length)
                message = json.loads(data.decode('utf-8'))
                
                # Process response
                self._process_response(message)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"TCP receiver error: {e}")
    
    def _run_udp_receiver(self):
        """Run UDP receiver loop"""
        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(4096)
                message = json.loads(data.decode('utf-8'))
                self._process_response(message)
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logger.error(f"UDP receiver error: {e}")
    
    def _process_response(self, message: Dict[str, Any]):
        """Process response from server"""
        receive_time = time.time()
        
        # Calculate round-trip latency
        if 'request_timestamp' in message:
            latency = (receive_time - message['request_timestamp']) * 1000  # ms
            self.stats.add_latency(latency)
            
            # Remove from pending
            msg_id = message.get('message_id', '').replace('_response', '')
            if msg_id in self.pending_messages:
                del self.pending_messages[msg_id]
            
            logger.debug(f"Received response, RTT: {latency:.1f}ms")
        
        # Add to response queue
        self.response_queue.put(message)
    
    def send_message(self, message_type: str, payload: Dict[str, Any],
                    sender_id: str = "client", receiver_id: str = "server") -> str:
        """Send a message to the server"""
        # Create message
        message = {
            "message_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "message_type": message_type,
            "payload": payload,
            "sender_id": sender_id,
            "receiver_id": receiver_id
        }
        
        # Track pending message
        self.pending_messages[message['message_id']] = message['timestamp']
        self.stats.messages_sent += 1
        
        # Send based on protocol
        if self.protocol == "tcp":
            self._send_tcp(message)
        else:
            self._send_udp(message)
        
        return message['message_id']
    
    def _send_tcp(self, message: Dict[str, Any]):
        """Send message via TCP"""
        if self.tcp_writer:
            try:
                data = json.dumps(message).encode('utf-8')
                length_data = struct.pack("!I", len(data))
                
                # Schedule write in async loop
                if self.event_loop:
                    asyncio.run_coroutine_threadsafe(
                        self._async_tcp_send(length_data + data),
                        self.event_loop
                    )
            except Exception as e:
                logger.error(f"TCP send error: {e}")
                self.stats.packet_loss += 1
    
    async def _async_tcp_send(self, data: bytes):
        """Async TCP send"""
        self.tcp_writer.write(data)
        await self.tcp_writer.drain()
    
    def _send_udp(self, message: Dict[str, Any]):
        """Send message via UDP"""
        if self.udp_socket:
            try:
                data = json.dumps(message).encode('utf-8')
                self.udp_socket.sendto(data, (self.server_host, self.server_port))
            except Exception as e:
                logger.error(f"UDP send error: {e}")
                self.stats.packet_loss += 1
    
    def get_response(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get a response from the queue"""
        timeout = timeout or self.timeout
        try:
            return self.response_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def ping(self) -> Optional[float]:
        """Send ping and measure latency"""
        msg_id = self.send_message("ping", {"ping": True})
        
        # Wait for response
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                response = self.response_queue.get(timeout=0.1)
                if response.get('message_id', '').startswith(msg_id):
                    latency = (time.time() - start_time) * 1000
                    return latency
            except queue.Empty:
                continue
        
        # Timeout
        self.stats.packet_loss += 1
        return None
    
    def stop(self):
        """Stop the client"""
        self.running = False
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.udp_socket:
            self.udp_socket.close()
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and int64 to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj
    
    def get_stats(self) -> Dict[str, Any]:
        """Get network statistics"""
        return {
            "messages_sent": self.stats.messages_sent,
            "messages_received": self.stats.messages_received,
            "packet_loss_rate": self.stats.get_packet_loss_rate(),
            "average_latency_ms": self.stats.get_average_latency(),
            "min_latency_ms": self.stats.min_latency if self.stats.min_latency != float('inf') else 0,
            "max_latency_ms": self.stats.max_latency,
            "pending_messages": len(self.pending_messages)
        }


class RealNetworkMessageHandler(MessageHandler):
    """
    Message handler that forwards messages through real network
    """
    
    def __init__(self, client: RealNetworkClient, node_id: str):
        self.client = client
        self.node_id = node_id
        self.received_messages: List[TimedMessage] = []
        
    def handle_message(self, message: TimedMessage) -> None:
        """Forward message through real network"""
        # Make payload JSON serializable
        serializable_payload = self.client._make_json_serializable(message.payload)
        
        # Send to real server
        response_id = self.client.send_message(
            message_type=message.message_type,
            payload=serializable_payload,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id
        )
        
        # Wait for response with short timeout to not block simulation
        response = self.client.get_response(timeout=0.05)
        
        if response and 'payload' in response:
            # The server should echo back the message with added delay
            # Store it as if it came through the network
            response_payload = response['payload']
            
            # Create timed message that looks like the original but with network delay
            response_msg = TimedMessage(
                payload=response_payload,  # This contains the original observation/action
                send_time=message.send_time,
                receive_time=time.time(),
                sender_id=message.sender_id,
                receiver_id=message.receiver_id,
                message_type=message.message_type
            )
            self.received_messages.append(response_msg)
    
    def clear(self) -> None:
        """Clear received messages"""
        self.received_messages.clear()
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy arrays and int64 to JSON-serializable types"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        else:
            return obj


def test_real_network_connection(server_host: str = "127.0.0.1",
                               server_port: int = 8765,
                               protocol: str = "tcp",
                               num_pings: int = 10) -> Dict[str, Any]:
    """
    Test connection to real network server
    """
    print(f"\nTesting connection to {server_host}:{server_port} ({protocol})...")
    
    client = RealNetworkClient(server_host, server_port, protocol)
    client.start()
    
    # Wait for connection
    time.sleep(0.5)
    
    # Send pings
    latencies = []
    for i in range(num_pings):
        latency = client.ping()
        if latency:
            latencies.append(latency)
            print(f"  Ping {i+1}: {latency:.1f}ms")
        else:
            print(f"  Ping {i+1}: timeout")
        time.sleep(0.1)
    
    # Get statistics
    stats = client.get_stats()
    client.stop()
    
    if latencies:
        import statistics
        stats['latency_stats'] = {
            'mean': statistics.mean(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'min': min(latencies),
            'max': max(latencies)
        }
    
    return stats


if __name__ == "__main__":
    # Test the client
    import sys
    
    if len(sys.argv) > 1:
        host = sys.argv[1]
    else:
        host = "127.0.0.1"
    
    # Test TCP
    print("Testing TCP connection...")
    tcp_stats = test_real_network_connection(host, 8765, "tcp")
    print(f"\nTCP Stats: {json.dumps(tcp_stats, indent=2)}")
    
    # Test UDP
    print("\nTesting UDP connection...")
    udp_stats = test_real_network_connection(host, 8766, "udp")
    print(f"\nUDP Stats: {json.dumps(udp_stats, indent=2)}")