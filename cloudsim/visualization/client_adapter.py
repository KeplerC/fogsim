import logging
import time
import uuid
import numpy as np
import socketio
import threading
import base64
from io import BytesIO
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)

class VisualizationClientAdapter:
    """Client adapter for connecting simulators to the visualization server."""
    
    def __init__(self, server_url='http://localhost:5000', simulation_id=None, simulator_type=None):
        """
        Initialize the visualization client adapter.
        
        Args:
            server_url: URL of the visualization server
            simulation_id: Unique identifier for this simulation instance
            simulator_type: Type of simulator (e.g., 'gym', 'carla')
        """
        self.server_url = server_url
        self.simulation_id = simulation_id or str(uuid.uuid4())
        self.simulator_type = simulator_type or 'unknown'
        
        # Create Socket.IO client
        self.sio = socketio.Client()
        self.connected = False
        self.registered = False
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Command handlers
        self.command_handlers = {}
        
        logger.info(f"VisualizationClientAdapter initialized for {self.simulation_id} ({self.simulator_type})")
    
    def _setup_event_handlers(self):
        """Set up Socket.IO event handlers."""
        
        @self.sio.event
        def connect():
            logger.info("Connected to visualization server")
            self.connected = True
            self._register_simulation()
        
        @self.sio.event
        def disconnect():
            logger.info("Disconnected from visualization server")
            self.connected = False
            self.registered = False
        
        @self.sio.event
        def connection_established(data):
            logger.info(f"Connection established: {data}")
        
        @self.sio.event
        def registration_confirmed(data):
            logger.info(f"Registration confirmed: {data}")
            self.registered = True
        
        @self.sio.event
        def error(data):
            logger.error(f"Error from server: {data}")
        
        @self.sio.event
        def command(data):
            """Handle command from the visualization server."""
            command_name = data.get('command')
            params = data.get('params', {})
            
            logger.info(f"Received command: {command_name} with params: {params}")
            
            # Execute the command handler if it exists
            if command_name in self.command_handlers:
                try:
                    result = self.command_handlers[command_name](params)
                    self.sio.emit('command_result', {
                        'command': command_name,
                        'result': result,
                        'success': True
                    })
                except Exception as e:
                    logger.error(f"Error executing command {command_name}: {str(e)}")
                    self.sio.emit('command_result', {
                        'command': command_name,
                        'error': str(e),
                        'success': False
                    })
            else:
                logger.warning(f"No handler for command: {command_name}")
                self.sio.emit('command_result', {
                    'command': command_name,
                    'error': f"Command not supported: {command_name}",
                    'success': False
                })
    
    def connect(self):
        """Connect to the visualization server."""
        if not self.connected:
            try:
                logger.info(f"Connecting to visualization server at {self.server_url}")
                self.sio.connect(self.server_url)
                return True
            except Exception as e:
                logger.error(f"Failed to connect to visualization server: {str(e)}")
                return False
        return True
    
    def disconnect(self):
        """Disconnect from the visualization server."""
        if self.connected:
            logger.info("Disconnecting from visualization server")
            self.sio.disconnect()
    
    def _register_simulation(self):
        """Register this client as a simulation with the server."""
        if self.connected and not self.registered:
            logger.info(f"Registering simulation {self.simulation_id} ({self.simulator_type})")
            self.sio.emit('register_simulation', {
                'simulation_id': self.simulation_id,
                'simulator_type': self.simulator_type
            })
    
    def send_frame(self, frame, timestamp=None):
        """
        Send a frame to the visualization server.
        
        Args:
            frame: Frame data (numpy array, PIL image, or encoded image string)
            timestamp: Frame timestamp (if None, current time is used)
        
        Returns:
            bool: True if frame was sent, False otherwise
        """
        if not self.connected or not self.registered:
            logger.warning("Cannot send frame - not connected or registered")
            return False
        
        if timestamp is None:
            timestamp = time.time()
            
        # Convert frame to base64 string if it's a numpy array or PIL image
        frame_data = self._convert_frame(frame)
        
        logger.debug(f"Sending frame to visualization server")
        self.sio.emit('frame', {
            'frame': frame_data,
            'timestamp': timestamp
        })
        
        return True
    
    def send_metrics(self, metrics, timestamp=None):
        """
        Send metrics to the visualization server.
        
        Args:
            metrics: Dict of metrics to send
            timestamp: Metrics timestamp (if None, current time is used)
        
        Returns:
            bool: True if metrics were sent, False otherwise
        """
        if not self.connected or not self.registered:
            logger.warning("Cannot send metrics - not connected or registered")
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        logger.debug(f"Sending metrics to visualization server")
        self.sio.emit('metrics', {
            'metrics': metrics,
            'timestamp': timestamp
        })
        
        return True
    
    def _convert_frame(self, frame):
        """Convert frame to a format suitable for transmission."""
        # If frame is already a string (e.g., base64 encoded), return as is
        if isinstance(frame, str):
            return frame
        
        # If frame is a numpy array
        if isinstance(frame, np.ndarray):
            # Convert to PIL Image
            if frame.ndim == 3 and frame.shape[2] == 3:
                # RGB image
                img = Image.fromarray(frame.astype('uint8'))
            else:
                # Grayscale or other format
                logger.warning(f"Converting unusual frame shape: {frame.shape}")
                img = Image.fromarray(frame.astype('uint8'))
        elif hasattr(frame, 'save'):
            # If frame is already a PIL Image
            img = frame
        else:
            logger.error(f"Unsupported frame type: {type(frame)}")
            return None
        
        # Convert to base64 string
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"
    
    def register_command_handler(self, command_name, handler_function):
        """
        Register a handler function for a specific command.
        
        Args:
            command_name: Name of the command to handle
            handler_function: Function to call when command is received. 
                             Should accept params dict as argument.
        """
        logger.info(f"Registering handler for command: {command_name}")
        self.command_handlers[command_name] = handler_function
    
    def run_in_background(self):
        """Start this client in a background thread."""
        def _run():
            self.connect()
            while self.connected:
                time.sleep(0.1)
        
        client_thread = threading.Thread(target=_run)
        client_thread.daemon = True
        client_thread.start()
        return client_thread 