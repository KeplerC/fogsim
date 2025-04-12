import time
import numpy as np
from cloudsim.network.nspy_simulator import NSPyNetworkSimulator

def test_network_simulator():
    """Simple test for the NSPyNetworkSimulator."""
    print("Testing NSPyNetworkSimulator...")
    
    # Create simulator with a moderate data rate
    simulator = NSPyNetworkSimulator(
        source_rate=10000.0,  # 10 Kbps
        weights=[1, 2],      # Weight flow 1 higher than flow 0
        debug=True
    )
    
    # Send messages from two different flows
    messages_sent = []
    
    # Send messages from flow 0 (default priority)
    for i in range(5):
        message = f"Flow 0 message {i}"
        msg_id = simulator.register_packet(message, flow_id=0, size=1000.0)
        messages_sent.append((msg_id, message, 0))
        print(f"Sent: {message} (ID: {msg_id}, Flow: 0)")
    
    # Send messages from flow 1 (higher priority)
    for i in range(5):
        message = f"Flow 1 message {i}"
        msg_id = simulator.register_packet(message, flow_id=1, size=1000.0)
        messages_sent.append((msg_id, message, 1))
        print(f"Sent: {message} (ID: {msg_id}, Flow: 1)")
    
    # Advance time and process messages
    print("\nAdvancing time to 0.5 seconds...")
    simulator.run_until(0.5)
    
    # Check for ready messages
    ready_messages = simulator.get_ready_messages()
    print(f"Ready messages at 0.5s: {ready_messages}")
    
    # Advance time further
    print("\nAdvancing time to 1.5 seconds...")
    simulator.run_until(1.5)
    
    # Check for more ready messages
    ready_messages = simulator.get_ready_messages()
    print(f"Ready messages at 1.5s: {ready_messages}")
    
    # Advance to ensure all messages are processed
    print("\nAdvancing time to 5.0 seconds...")
    simulator.run_until(5.0)
    
    # Check final messages
    ready_messages = simulator.get_ready_messages()
    print(f"Ready messages at 5.0s: {ready_messages}")
    
    # Print any pending messages
    print("\nPending messages:")
    for msg_id, (message, sent_time, flow_id) in simulator.pending_packets.items():
        print(f"  - {message} (ID: {msg_id}, Flow: {flow_id}, Sent at: {sent_time})")
    
    # Clean up
    simulator.close()
    print("Test completed.")

if __name__ == "__main__":
    test_network_simulator() 