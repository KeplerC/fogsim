#!/usr/bin/env python3

import sys
import os
import time
import threading
import argparse
import subprocess
import signal
import logging
import requests
from requests.exceptions import ConnectionError

# Add the parent directory to the path for importing the base adaptor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptor_base import AdaptorBase

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('test_pub_sub')

class TestAdaptor(AdaptorBase):
    """Test adaptor for pub/sub testing."""
    
    def __init__(self, adaptor_id, meta_simulator_url, poll_interval=0.5):
        super().__init__(adaptor_id=adaptor_id, 
                        meta_simulator_url=meta_simulator_url,
                        poll_interval=poll_interval)
        self.received_messages = {}
        self.msg_count = 0
        
    def handle_message(self, data, topic):
        """Handle a message received on a topic."""
        if topic not in self.received_messages:
            self.received_messages[topic] = []
        self.received_messages[topic].append(data)
        self.msg_count += 1
        logger.info(f"Adaptor {self.adaptor_id} received message on topic {topic}: {data}")
        
    def subscribe_to_topic(self, topic):
        """Subscribe to a topic."""
        self.subscribe(topic, self.handle_message)
        logger.info(f"Adaptor {self.adaptor_id} subscribed to topic: {topic}")
        
    def publish_message(self, topic, message):
        """Publish a message to a topic."""
        msg_id = self.publish(topic, message)
        logger.info(f"Adaptor {self.adaptor_id} published message to topic {topic}: {message}")
        return msg_id

def wait_for_service(url, max_retries=30, retry_interval=1.0):
    """Wait for a service to be available by polling the URL."""
    logger.info(f"Waiting for service at {url} to become available...")
    
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                logger.info(f"Service at {url} is available!")
                return True
        except ConnectionError:
            logger.debug(f"Service not available yet, retrying ({i+1}/{max_retries})...")
        except Exception as e:
            logger.debug(f"Error checking service: {str(e)}")
        
        time.sleep(retry_interval)
    
    logger.error(f"Service at {url} did not become available after {max_retries} attempts")
    return False

def start_meta_simulator():
    """Start the meta simulator as a subprocess."""
    logger.info("Starting meta simulator...")
    meta_simulator_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                       "meta_simulator", "meta_simulator.py")
    
    if not os.path.exists(meta_simulator_path):
        logger.error(f"Meta simulator not found at {meta_simulator_path}")
        return None
        
    # Start the meta simulator
    process = subprocess.Popen([sys.executable, meta_simulator_path], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    # Wait for the meta simulator to be ready to accept connections
    if not wait_for_service("http://localhost:5000"):
        logger.error("Failed to start meta simulator")
        process.terminate()
        return None
    
    logger.info("Meta simulator started and ready")
    return process

def run_test(meta_simulator_url="http://localhost:5000"):
    """Run the pub/sub test."""
    # Create publisher and subscriber adaptors
    publisher = TestAdaptor("test_publisher", meta_simulator_url)
    subscriber = TestAdaptor("test_subscriber", meta_simulator_url)
    
    # Start adaptors
    publisher.start()
    subscriber.start()
    
    # Subscribe to topics
    test_topics = ["test/topic1", "test/topic2", "test/topic3"]
    for topic in test_topics:
        subscriber.subscribe_to_topic(topic)
    
    # Wait a bit for subscriptions to be registered
    time.sleep(1)
    
    # Publish messages
    test_messages = {
        "test/topic1": ["Hello, world!", "Test message 1", "Another message"],
        "test/topic2": ["Topic 2 message", "Another topic 2 message"],
        "test/topic3": ["Topic 3 message"]
    }
    
    for topic, messages in test_messages.items():
        for message in messages:
            publisher.publish_message(topic, message)
            time.sleep(0.5)  # Small delay between messages
    
    # Wait for messages to be received
    logger.info("Waiting for messages to be received...")
    time.sleep(5)
    
    # Check results
    success = True
    expected_count = sum(len(messages) for messages in test_messages.values())
    
    if subscriber.msg_count != expected_count:
        logger.error(f"Expected {expected_count} messages, but received {subscriber.msg_count}")
        success = False
    
    for topic, messages in test_messages.items():
        if topic not in subscriber.received_messages:
            logger.error(f"No messages received for topic {topic}")
            success = False
            continue
            
        if len(subscriber.received_messages[topic]) != len(messages):
            logger.error(f"Expected {len(messages)} messages for topic {topic}, "
                       f"but received {len(subscriber.received_messages[topic])}")
            success = False
            
        # Check message content
        for i, msg in enumerate(messages):
            if i < len(subscriber.received_messages[topic]):
                received = subscriber.received_messages[topic][i]
                if received != msg:
                    logger.error(f"Expected '{msg}' but received '{received}' for topic {topic}")
                    success = False
    
    # Stop adaptors
    publisher.stop()
    subscriber.stop()
    
    return success

def main():
    parser = argparse.ArgumentParser(description='Test pub/sub through meta simulator')
    parser.add_argument('--meta-simulator-url', type=str, default="http://localhost:5000",
                       help='Meta simulator URL')
    parser.add_argument('--start-simulator', action='store_true',
                       help='Start the meta simulator as part of the test')
    
    args = parser.parse_args()
    
    meta_simulator_process = None
    if args.start_simulator:
        meta_simulator_process = start_meta_simulator()
        if not meta_simulator_process:
            logger.error("Failed to start meta simulator")
            return 1
    
    try:
        success = run_test(args.meta_simulator_url)
        
        if success:
            logger.info("Test passed! All messages were sent and received correctly.")
            return 0
        else:
            logger.error("Test failed. See errors above.")
            return 1
            
    finally:
        if meta_simulator_process:
            logger.info("Stopping meta simulator...")
            meta_simulator_process.terminate()
            try:
                meta_simulator_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                meta_simulator_process.kill()
            logger.info("Meta simulator stopped")

if __name__ == '__main__':
    sys.exit(main()) 