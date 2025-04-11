#!/usr/bin/env python3

import requests
import json
import time
import threading
import logging
import os
import uuid
import base64

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class AdaptorBase:
    """Base class for adaptors that communicate with the meta simulator."""

    def __init__(self, 
                 adaptor_id=None, 
                 meta_simulator_url=None, 
                 poll_interval=0.5):
        """
        Initialize the base adaptor.
        
        Args:
            adaptor_id: Unique identifier for this adaptor
            meta_simulator_url: URL of the meta simulator
            poll_interval: Interval in seconds for polling the meta simulator
        """
        # Initialize parameters
        self.adaptor_id = adaptor_id or os.environ.get('ADAPTOR_ID', f'adaptor_{uuid.uuid4().hex[:8]}')
        self.meta_simulator_url = meta_simulator_url or os.environ.get('META_SIMULATOR_URL', 'http://meta_simulator:5000')
        self.poll_interval = float(poll_interval or os.environ.get('POLL_INTERVAL', '0.5'))
        
        # Initialize state
        self.current_time = {'seconds': 0, 'nanoseconds': 0}
        self.registered = False
        self.running = True
        self.timing_data = {}  # message_id -> start_time
        
        # Subscription management
        self.subscriptions = {}  # topic -> callback
        
        # Start polling thread
        self._poll_thread = None
        self.logger = logging.getLogger(f'adaptor.{self.adaptor_id}')
        self.logger.info(f"Adaptor {self.adaptor_id} initialized")
        
    def start(self):
        """Start the adaptor and polling thread."""
        self.register_with_meta_simulator()
        
        if not self._poll_thread or not self._poll_thread.is_alive():
            self._poll_thread = threading.Thread(target=self._poll_meta_simulator)
            self._poll_thread.daemon = True
            self._poll_thread.start()
            
        self.logger.info(f"Adaptor {self.adaptor_id} started")
        
    def stop(self):
        """Stop the adaptor."""
        self.running = False
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
        self.logger.info(f"Adaptor {self.adaptor_id} stopped")
        
    def register_with_meta_simulator(self):
        """Register this adaptor with the meta simulator."""
        try:
            response = requests.post(
                f"{self.meta_simulator_url}/register",
                json={'adaptor_id': self.adaptor_id}
            )
            if response.status_code == 200 and response.json().get('success'):
                self.registered = True
                self.logger.info(f"Registered with meta simulator: {self.adaptor_id}")
            else:
                self.logger.error(f"Failed to register: {response.text}")
        except Exception as e:
            self.logger.error(f"Error registering with meta simulator: {str(e)}")
            
    def _poll_meta_simulator(self):
        """Poll the meta simulator for messages."""
        while self.running:
            if not self.registered:
                # Try to register again if not registered
                self.register_with_meta_simulator()
                time.sleep(1)
                continue
            
            try:
                # Get list of topics we're subscribed to
                topics = list(self.subscriptions.keys())
                
                # Poll for messages
                response = requests.post(
                    f"{self.meta_simulator_url}/poll",
                    json={
                        'topics': topics,
                        'last_update_time': self.current_time
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get('success'):
                        # Update current time
                        self.current_time = data['state']['current_time']
                        
                        # Process messages if any
                        messages = data.get('messages', [])
                        if messages:
                            self.logger.info(f"Received {len(messages)} messages to process")
                            for msg in messages:
                                self._process_message(msg)
                else:
                    self.logger.error(f"Failed to poll: {response.text}")
            
            except Exception as e:
                self.logger.error(f"Error polling meta simulator: {str(e)}")
            
            # Sleep for poll interval
            time.sleep(self.poll_interval)
            
    def _process_message(self, message):
        """Process an incoming message."""
        message_id = message.get('message_id', '')
        self.logger.info(f"Processing message: {message_id}")
        
        # Check if it's a simulator state message
        if 'simulator_state' in message:
            self._handle_simulator_state(message)
        
        # Check if it's an algorithm response message
        elif 'algorithm_response' in message:
            self._handle_algorithm_response(message)
        
        # Implement other message types as needed
            
    def _handle_simulator_state(self, message):
        """Handle simulator state message."""
        simulator_state = message.get('simulator_state', {})
        topic = simulator_state.get('frame_id', '')
        
        if topic in self.subscriptions:
            # Extract the message data
            if 'pickled_data_b64' in simulator_state:
                try:
                    import pickle
                    pickled_data_b64 = simulator_state.get('pickled_data_b64', '')
                    pickled_data = base64.b64decode(pickled_data_b64)
                    data = pickle.loads(pickled_data)
                    
                    # Call the callback with the data
                    callback = self.subscriptions[topic]
                    callback(data, topic)
                    
                except Exception as e:
                    self.logger.error(f"Failed to unpickle message data: {str(e)}")
        else:
            self.logger.warning(f"Received message for unsubscribed topic: {topic}")
            
    def _handle_algorithm_response(self, message):
        """Handle algorithm response message."""
        topic = message.get('topic', '')
        algorithm_response = message.get('algorithm_response', {})
        
        if topic in self.subscriptions:
            # Extract the message data
            if 'pickled_data_b64' in algorithm_response:
                try:
                    import pickle
                    pickled_data_b64 = algorithm_response.get('pickled_data_b64', '')
                    pickled_data = base64.b64decode(pickled_data_b64)
                    data = pickle.loads(pickled_data)
                    
                    # Call the callback with the data
                    callback = self.subscriptions[topic]
                    callback(data, topic)
                    
                except Exception as e:
                    self.logger.error(f"Failed to unpickle message data: {str(e)}")
        else:
            self.logger.warning(f"Received message for unsubscribed topic: {topic}")
            
    def subscribe(self, topic, callback):
        """
        Subscribe to a topic.
        
        Args:
            topic: The topic to subscribe to
            callback: The callback function to call when a message is received
        
        Returns:
            True if successful, False otherwise
        """
        self.subscriptions[topic] = callback
        self.logger.info(f"Subscribed to topic: {topic}")
        return True
        
    def unsubscribe(self, topic):
        """
        Unsubscribe from a topic.
        
        Args:
            topic: The topic to unsubscribe from
            
        Returns:
            True if successful, False otherwise
        """
        if topic in self.subscriptions:
            del self.subscriptions[topic]
            self.logger.info(f"Unsubscribed from topic: {topic}")
            return True
        return False
        
    def publish(self, topic, data, simulate_network=False):
        """
        Publish data to a topic.
        
        Args:
            topic: The topic to publish to
            data: The data to publish
            simulate_network: Whether to simulate network delay
            
        Returns:
            message_id if successful, None otherwise
        """
        try:
            # Pickle the data
            import pickle
            pickled_data = pickle.dumps(data)
            
            # Encode as base64
            b64_data = base64.b64encode(pickled_data).decode('ascii')
            
            # Create the message
            message = {
                'source_id': self.adaptor_id,
                'topic': topic,
                'timestamp': self.current_time,
                'simulate_network': simulate_network,
                'algorithm_response': {
                    'pickled_data_b64': b64_data,
                    'msg_type': data.__class__.__module__ + '.' + data.__class__.__name__,
                    'compute_time_ns': 0,
                    'status': 'success'
                }
            }
            
            # Send the message
            response = requests.post(
                f"{self.meta_simulator_url}/send",
                json=message
            )
            
            if response.status_code == 200 and response.json().get('success'):
                message_id = response.json().get('message_id')
                self.logger.info(f"Published message to topic {topic}: {message_id}")
                return message_id
            else:
                self.logger.error(f"Failed to publish message: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error publishing message: {str(e)}")
            return None

    def get_state(self):
        """Get the state of the meta simulator."""
        try:
            response = requests.get(f"{self.meta_simulator_url}/state")
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data.get('state')
            return None
        except Exception as e:
            self.logger.error(f"Error getting state: {str(e)}")
            return None 