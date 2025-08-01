"""
FogSim Real Network Server

This server runs on the network (local or remote) and handles simulation
messages, allowing measurement of real network latency and behavior.
"""

import asyncio
import json
import time
import logging
import argparse
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import socket
import struct

logger = logging.getLogger(__name__)


@dataclass
class SimulationMessage:
    """Message format for simulation communication"""
    message_id: str
    timestamp: float
    message_type: str  # "observation", "action", "ping"
    payload: Dict[str, Any]
    sender_id: str
    receiver_id: str


class FogSimServer:
    """
    Echo server for FogSim real network mode.
    
    This server simulates the remote simulation component,
    receiving observations and returning actions.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.clients = {}
        self.message_count = 0
        self.start_time = time.time()
        
    async def handle_client(self, reader: asyncio.StreamReader, 
                          writer: asyncio.StreamWriter):
        """Handle a client connection"""
        client_addr = writer.get_extra_info('peername')
        logger.info(f"New client connected: {client_addr}")
        
        try:
            while True:
                # Read message length (4 bytes)
                length_data = await reader.readexactly(4)
                msg_length = struct.unpack("!I", length_data)[0]
                
                # Read message data
                data = await reader.readexactly(msg_length)
                
                # Process message
                response = await self.process_message(data, client_addr)
                
                # Send response
                response_data = json.dumps(response).encode()
                writer.write(struct.pack("!I", len(response_data)))
                writer.write(response_data)
                await writer.drain()
                
        except asyncio.IncompleteReadError:
            logger.info(f"Client {client_addr} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_addr}: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def process_message(self, data: bytes, client_addr) -> Dict[str, Any]:
        """Process incoming message and generate response"""
        try:
            # Decode message
            message_str = data.decode('utf-8')
            message_data = json.loads(message_str)
            
            # Track statistics
            self.message_count += 1
            receive_time = time.time()
            
            # Create message object
            msg = SimulationMessage(**message_data)
            
            # Log message info
            latency = (receive_time - msg.timestamp) * 1000  # ms
            logger.debug(f"Received {msg.message_type} from {client_addr}, "
                        f"latency: {latency:.1f}ms")
            
            # Process based on message type
            if msg.message_type == "ping":
                # Simple ping response
                response_payload = {
                    "pong": True,
                    "server_time": receive_time,
                    "message_count": self.message_count
                }
                
            elif msg.message_type == "observation":
                # Echo back the observation (simulating network delay)
                # In a real deployment, this would be processed by a remote controller
                response_payload = msg.payload  # Echo the entire payload
                
                # Simulate processing delay
                await asyncio.sleep(0.001)
                
            elif msg.message_type == "action":
                # Echo back the action (simulating network delay)
                # In a real deployment, this would be sent to the robot
                response_payload = msg.payload  # Echo the entire payload
                
            else:
                # For any other message type, echo back the payload
                response_payload = msg.payload
            
            # Create response
            response = {
                "message_id": msg.message_id + "_response",
                "timestamp": time.time(),
                "message_type": msg.message_type + "_response",
                "payload": response_payload,
                "sender_id": "server",
                "receiver_id": msg.sender_id,
                "request_timestamp": msg.timestamp,
                "server_receive_time": receive_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def start(self):
        """Start the server"""
        server = await asyncio.start_server(
            self.handle_client, self.host, self.port
        )
        
        addr = server.sockets[0].getsockname()
        logger.info(f"FogSim server running on {addr[0]}:{addr[1]}")
        print(f"\nFogSim Real Network Server")
        print(f"Listening on {addr[0]}:{addr[1]}")
        print(f"Ready to accept connections...\n")
        
        async with server:
            await server.serve_forever()


class FogSimUDPServer:
    """
    UDP version for lower latency measurements
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8766):
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, port))
        self.message_count = 0
        
    def run(self):
        """Run the UDP server"""
        print(f"\nFogSim UDP Server")
        print(f"Listening on {self.host}:{self.port}")
        print(f"Ready for UDP packets...\n")
        
        while True:
            try:
                # Receive data
                data, addr = self.sock.recvfrom(4096)
                receive_time = time.time()
                
                # Decode message
                message_data = json.loads(data.decode('utf-8'))
                msg = SimulationMessage(**message_data)
                
                # Calculate latency
                latency = (receive_time - msg.timestamp) * 1000
                self.message_count += 1
                
                logger.debug(f"UDP: {msg.message_type} from {addr}, "
                           f"latency: {latency:.1f}ms")
                
                # Create response
                response = {
                    "message_id": msg.message_id + "_response",
                    "timestamp": time.time(),
                    "message_type": msg.message_type + "_response",
                    "payload": {"echo": True, "count": self.message_count},
                    "sender_id": "server",
                    "receiver_id": msg.sender_id,
                    "request_timestamp": msg.timestamp,
                    "latency_ms": latency
                }
                
                # Send response
                response_data = json.dumps(response).encode('utf-8')
                self.sock.sendto(response_data, addr)
                
            except Exception as e:
                logger.error(f"UDP error: {e}")


def main():
    """Run the FogSim server"""
    parser = argparse.ArgumentParser(description="FogSim Real Network Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port for TCP server")
    parser.add_argument("--udp-port", type=int, default=8766, help="Port for UDP server")
    parser.add_argument("--protocol", choices=["tcp", "udp", "both"], default="tcp",
                       help="Protocol to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.protocol == "tcp" or args.protocol == "both":
        # Run TCP server
        server = FogSimServer(args.host, args.port)
        
        if args.protocol == "tcp":
            asyncio.run(server.start())
        else:
            # Run TCP in thread for 'both' mode
            import threading
            tcp_thread = threading.Thread(
                target=lambda: asyncio.run(server.start()),
                daemon=True
            )
            tcp_thread.start()
    
    if args.protocol == "udp" or args.protocol == "both":
        # Run UDP server
        udp_server = FogSimUDPServer(args.host, args.udp_port)
        udp_server.run()


if __name__ == "__main__":
    main()