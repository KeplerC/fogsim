"""
Simple Echo Server for Real Network Mode Testing

This server receives messages and echoes them back to measure real network round-trip times.
"""

import socket
import threading
import time
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class EchoServer:
    """Simple TCP echo server for network latency testing."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.thread = None
        
    def start(self) -> None:
        """Start the echo server in a background thread."""
        if self.running:
            return
            
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((self.host, self.port))
            
            # If port was 0, get the actual assigned port
            if self.port == 0:
                self.port = self.socket.getsockname()[1]
            
            self.socket.listen(5)
            self.running = True
            
            self.thread = threading.Thread(target=self._run_server, daemon=True)
            self.thread.start()
            
            logger.info(f"Echo server started on {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start echo server: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the echo server."""
        if not self.running:
            return
            
        self.running = False
        if self.socket:
            self.socket.close()
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            
        logger.info("Echo server stopped")
    
    def _run_server(self) -> None:
        """Main server loop."""
        while self.running:
            try:
                client_socket, address = self.socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address),
                    daemon=True
                )
                client_thread.start()
                
            except OSError:
                # Socket was closed
                break
            except Exception as e:
                logger.error(f"Error accepting connection: {e}")
    
    def _handle_client(self, client_socket: socket.socket, address) -> None:
        """Handle a single client connection."""
        logger.debug(f"Client connected from {address}")
        
        try:
            while self.running:
                # Receive message
                data = client_socket.recv(4096)
                if not data:
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(data.decode('utf-8'))
                    
                    # Add server timestamp
                    message['server_timestamp'] = time.time()
                    
                    # Echo back the message
                    response = json.dumps(message).encode('utf-8')
                    client_socket.send(response)
                    
                    logger.debug(f"Echoed message: {message.get('id', 'unknown')}")
                    
                except json.JSONDecodeError:
                    # Invalid JSON, just echo raw data
                    client_socket.send(data)
                    
        except Exception as e:
            logger.debug(f"Client {address} disconnected: {e}")
        finally:
            client_socket.close()


class EchoClient:
    """Client for communicating with echo server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8888):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to the echo server."""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)  # 5 second timeout
            self.socket.connect((self.host, self.port))
            self.connected = True
            logger.debug(f"Connected to echo server at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to connect to echo server: {e}")
            return False
    
    def send_message(self, message: dict) -> Optional[dict]:
        """Send message and wait for echo response."""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Add client timestamp
            message['client_send_time'] = time.time()
            
            # Send message (handle numpy types)
            data = json.dumps(message, default=self._json_serializer).encode('utf-8')
            self.socket.send(data)
            
            # Receive response
            response_data = self.socket.recv(4096)
            response = json.loads(response_data.decode('utf-8'))
            
            # Calculate round-trip time
            response['client_receive_time'] = time.time()
            response['round_trip_time'] = response['client_receive_time'] - response['client_send_time']
            
            logger.debug(f"Round-trip time: {response['round_trip_time']*1000:.1f}ms")
            return response
            
        except Exception as e:
            logger.warning(f"Error sending message: {e}")
            self.connected = False
            return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types."""
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def close(self) -> None:
        """Close connection."""
        if self.socket:
            self.socket.close()
            self.connected = False


def start_echo_server(host: str = "127.0.0.1", port: int = 8888) -> EchoServer:
    """Start an echo server and return the instance."""
    server = EchoServer(host, port)
    server.start()
    return server


if __name__ == "__main__":
    # Test the echo server
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        # Run as server
        server = start_echo_server()
        print(f"Echo server running on 127.0.0.1:8888")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.stop()
            print("\nServer stopped")
    
    else:
        # Run as client test
        print("Testing echo server...")
        
        # Start server
        server = start_echo_server()
        time.sleep(0.1)  # Let server start
        
        # Test client
        client = EchoClient()
        
        for i in range(5):
            message = {"id": f"test_{i}", "data": f"Hello {i}"}
            response = client.send_message(message)
            
            if response:
                rtt = response['round_trip_time'] * 1000
                print(f"Message {i}: RTT = {rtt:.1f}ms")
            else:
                print(f"Message {i}: Failed")
        
        client.close()
        server.stop()