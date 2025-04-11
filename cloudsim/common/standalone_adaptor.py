#!/usr/bin/env python3

import sys
import os
import logging
import time
import pickle
import base64
import threading
import argparse

# Add the parent directory to the path for importing the base adaptor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.adaptor_base import AdaptorBase

class StandaloneAdaptor(AdaptorBase):
    """Standalone adaptor for testing that doesn't require ROS."""
    
    def __init__(self, 
                adaptor_id=None,
                meta_simulator_url=None,
                poll_interval=0.5):
        """
        Initialize the standalone adaptor.
        
        Args:
            adaptor_id: Unique identifier for this adaptor
            meta_simulator_url: URL of the meta simulator
            poll_interval: Interval in seconds for polling the meta simulator
        """
        # Initialize adaptor base
        super().__init__(
            adaptor_id=adaptor_id, 
            meta_simulator_url=meta_simulator_url,
            poll_interval=poll_interval
        )
        
        self.logger.info(f"Standalone Adaptor {self.adaptor_id} initialized")
        
    def start_interactive(self):
        """Start the adaptor and run an interactive shell."""
        self.start()
        self.logger.info("Interactive shell started. Enter commands to interact with the adaptor.")
        self.logger.info("Available commands:")
        self.logger.info("  - sub <topic> : Subscribe to a topic")
        self.logger.info("  - unsub <topic> : Unsubscribe from a topic")
        self.logger.info("  - pub <topic> <message> : Publish a message to a topic")
        self.logger.info("  - list : List all topics")
        self.logger.info("  - state : Get the state of the meta simulator")
        self.logger.info("  - exit : Exit the interactive shell")
        
        while True:
            try:
                cmd = input("> ")
                parts = cmd.strip().split()
                
                if not parts:
                    continue
                
                if parts[0] == "exit":
                    break
                elif parts[0] == "sub" and len(parts) > 1:
                    topic = parts[1]
                    self.subscribe(topic, self.handle_message)
                    print(f"Subscribed to {topic}")
                elif parts[0] == "unsub" and len(parts) > 1:
                    topic = parts[1]
                    self.unsubscribe(topic)
                    print(f"Unsubscribed from {topic}")
                elif parts[0] == "pub" and len(parts) > 2:
                    topic = parts[1]
                    message = " ".join(parts[2:])
                    msg_id = self.publish(topic, message)
                    if msg_id:
                        print(f"Published message {msg_id} to {topic}")
                    else:
                        print(f"Failed to publish message to {topic}")
                elif parts[0] == "list":
                    print(f"Subscribed topics: {list(self.subscriptions.keys())}")
                elif parts[0] == "state":
                    state = self.get_state()
                    print(f"Meta simulator state: {state}")
                else:
                    print(f"Unknown command: {parts[0]}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {str(e)}")
        
        self.stop()
        
    def handle_message(self, data, topic):
        """Handle a message received on a topic."""
        print(f"Received message on topic {topic}: {data}")
        
def main():
    parser = argparse.ArgumentParser(description='Standalone adaptor for meta simulator')
    parser.add_argument('--adaptor-id', type=str, help='Adaptor ID')
    parser.add_argument('--meta-simulator-url', type=str, help='Meta simulator URL')
    parser.add_argument('--poll-interval', type=float, help='Poll interval in seconds')
    
    args = parser.parse_args()
    
    adaptor = StandaloneAdaptor(
        adaptor_id=args.adaptor_id,
        meta_simulator_url=args.meta_simulator_url,
        poll_interval=args.poll_interval
    )
    
    try:
        adaptor.start_interactive()
    except KeyboardInterrupt:
        pass
    finally:
        adaptor.stop()

if __name__ == '__main__':
    main() 