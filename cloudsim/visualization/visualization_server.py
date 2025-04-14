import os
import logging
import numpy as np
import time
import json
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import threading
import base64
from io import BytesIO
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationServer:
    """Server that handles visualization for cloudsim co-simulators."""
    
    def __init__(self, host='0.0.0.0', port=5000, static_folder='static', template_folder='templates'):
        """
        Initialize the visualization server.
        
        Args:
            host: Host address to bind the server to
            port: Port to listen on
            static_folder: Folder for static files
            template_folder: Folder for templates
        """
        self.host = host
        self.port = port
        
        # Create Flask app and SocketIO instance
        self.app = Flask(__name__, 
                        static_folder=static_folder,
                        template_folder=template_folder)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Track connected clients
        self.clients = {}
        
        # Store the most recent data for each simulation
        self.current_frames = {}
        self.metrics = {}
        
        # Set up routes and event handlers
        self._setup_routes()
        self._setup_socket_events()
        
        logger.info(f"Visualization server initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Set up HTTP routes for the Flask app."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/simulations', methods=['GET'])
        def get_simulations():
            """Get a list of all connected simulations."""
            return jsonify(list(self.clients.keys()))
        
        @self.app.route('/api/metrics/<simulation_id>', methods=['GET'])
        def get_metrics(simulation_id):
            """Get the latest metrics for a simulation."""
            if simulation_id in self.metrics:
                return jsonify(self.metrics[simulation_id])
            return jsonify({"error": "Simulation not found"}), 404
    
    def _setup_socket_events(self):
        """Set up Socket.IO event handlers."""
        
        @self.socketio.on('connect')
        def on_connect():
            """Handle client connection."""
            client_id = request.sid
            logger.info(f"Client connected: {client_id}")
            
            # Add client to tracking
            self.clients[client_id] = {
                'connected_at': time.time(),
                'type': 'unknown',
                'simulation_id': None
            }
            
            # Notify client of successful connection
            emit('connection_established', {'client_id': client_id})
        
        @self.socketio.on('disconnect')
        def on_disconnect():
            """Handle client disconnection."""
            client_id = request.sid
            logger.info(f"Client disconnected: {client_id}")
            
            # Remove client from tracking
            if client_id in self.clients:
                simulation_id = self.clients[client_id].get('simulation_id')
                if simulation_id:
                    # Clean up stored data for this simulation
                    self.current_frames.pop(simulation_id, None)
                    self.metrics.pop(simulation_id, None)
                
                del self.clients[client_id]
                
                # Notify other clients about disconnection
                emit('simulation_disconnected', {'simulation_id': simulation_id}, broadcast=True)
        
        @self.socketio.on('register_simulation')
        def register_simulation(data):
            """Register a client as a simulation."""
            client_id = request.sid
            simulation_id = data.get('simulation_id')
            simulator_type = data.get('simulator_type', 'unknown')
            
            if not simulation_id:
                emit('error', {'message': 'Missing simulation_id'})
                return
            
            logger.info(f"Registering simulation: {simulation_id} ({simulator_type})")
            
            # Update client information
            self.clients[client_id].update({
                'type': 'simulation',
                'simulation_id': simulation_id,
                'simulator_type': simulator_type
            })
            
            # Initialize data storage for this simulation
            if simulation_id not in self.current_frames:
                self.current_frames[simulation_id] = None
            if simulation_id not in self.metrics:
                self.metrics[simulation_id] = {}
            
            # Notify client of successful registration
            emit('registration_confirmed', {'simulation_id': simulation_id})
            
            # Notify other clients about new simulation
            emit('simulation_connected', {
                'simulation_id': simulation_id,
                'simulator_type': simulator_type
            }, broadcast=True)
        
        @self.socketio.on('register_viewer')
        def register_viewer(data):
            """Register a client as a viewer."""
            client_id = request.sid
            viewer_id = data.get('viewer_id')
            
            if not viewer_id:
                emit('error', {'message': 'Missing viewer_id'})
                return
            
            logger.info(f"Registering viewer: {viewer_id}")
            
            # Update client information
            self.clients[client_id].update({
                'type': 'viewer',
                'viewer_id': viewer_id
            })
            
            # Send list of active simulations to the viewer
            active_simulations = []
            for cid, client in self.clients.items():
                if client.get('type') == 'simulation':
                    active_simulations.append({
                        'simulation_id': client.get('simulation_id'),
                        'simulator_type': client.get('simulator_type')
                    })
            
            emit('active_simulations', {'simulations': active_simulations})
        
        @self.socketio.on('frame')
        def receive_frame(data):
            """Receive and process a frame from a simulation."""
            client_id = request.sid
            simulation_id = self.clients.get(client_id, {}).get('simulation_id')
            
            if not simulation_id:
                emit('error', {'message': 'Unregistered client cannot send frames'})
                return
            
            frame_data = data.get('frame')
            timestamp = data.get('timestamp', time.time())
            
            logger.debug(f"Received frame from {simulation_id} at {timestamp}")
            
            # Store the frame
            self.current_frames[simulation_id] = {
                'data': frame_data,
                'timestamp': timestamp
            }
            
            # Forward frame to all connected viewers
            for cid, client in self.clients.items():
                if client.get('type') == 'viewer':
                    emit('frame_update', {
                        'simulation_id': simulation_id,
                        'frame': frame_data,
                        'timestamp': timestamp
                    }, room=cid)
        
        @self.socketio.on('metrics')
        def receive_metrics(data):
            """Receive and process metrics from a simulation."""
            client_id = request.sid
            simulation_id = self.clients.get(client_id, {}).get('simulation_id')
            
            if not simulation_id:
                emit('error', {'message': 'Unregistered client cannot send metrics'})
                return
            
            metrics = data.get('metrics', {})
            timestamp = data.get('timestamp', time.time())
            
            logger.debug(f"Received metrics from {simulation_id} at {timestamp}")
            
            # Store the metrics
            self.metrics[simulation_id] = {
                'data': metrics,
                'timestamp': timestamp
            }
            
            # Forward metrics to all connected viewers
            for cid, client in self.clients.items():
                if client.get('type') == 'viewer':
                    emit('metrics_update', {
                        'simulation_id': simulation_id,
                        'metrics': metrics,
                        'timestamp': timestamp
                    }, room=cid)
        
        @self.socketio.on('command')
        def receive_command(data):
            """Receive and forward a command from a viewer to a simulation."""
            client_id = request.sid
            
            # Verify the client is a viewer
            if self.clients.get(client_id, {}).get('type') != 'viewer':
                emit('error', {'message': 'Only viewers can send commands'})
                return
            
            target_simulation = data.get('simulation_id')
            command = data.get('command')
            params = data.get('params', {})
            
            if not target_simulation or not command:
                emit('error', {'message': 'Missing required command parameters'})
                return
            
            logger.info(f"Received command from viewer: {command} to {target_simulation}")
            
            # Find the target simulation client
            target_client_id = None
            for cid, client in self.clients.items():
                if (client.get('type') == 'simulation' and 
                    client.get('simulation_id') == target_simulation):
                    target_client_id = cid
                    break
            
            if target_client_id:
                # Forward command to the simulation
                emit('command', {
                    'command': command,
                    'params': params
                }, room=target_client_id)
                
                # Acknowledge command to sender
                emit('command_sent', {
                    'simulation_id': target_simulation,
                    'command': command,
                    'params': params
                })
            else:
                emit('error', {'message': f'Simulation {target_simulation} not found'})
    
    def run(self, debug=False):
        """Run the visualization server."""
        logger.info(f"Starting visualization server on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)
        
    def run_in_thread(self):
        """Run the server in a separate thread."""
        server_thread = threading.Thread(target=self.run)
        server_thread.daemon = True
        server_thread.start()
        return server_thread

if __name__ == "__main__":
    # Create and run the visualization server
    server = VisualizationServer()
    server.run(debug=True) 