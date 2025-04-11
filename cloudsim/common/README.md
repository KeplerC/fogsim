# Meta Simulator Adaptor Framework

This framework provides a clean and flexible way to communicate with the Meta Simulator without hardcoding topic names or communication patterns.

## Components

### 1. AdaptorBase

The base class for all adaptors that communicate with the Meta Simulator. It provides:

- Registration with the Meta Simulator
- Topic-based pub/sub functionality
- Polling for messages
- Message processing

### 2. ROSAdaptor

A ROS-specific adaptor that bridges between ROS topics and the Meta Simulator. It:

- Auto-discovers ROS topics
- Creates appropriate publishers and subscribers
- Routes messages between ROS and the Meta Simulator

### 3. StandaloneAdaptor

A simple standalone adaptor for testing and debugging, with an interactive shell.

## Usage

### Using the Base Adaptor

```python
from common.adaptor_base import AdaptorBase

# Create an adaptor
adaptor = AdaptorBase(
    adaptor_id="my_adaptor",
    meta_simulator_url="http://meta_simulator:5000"
)

# Start the adaptor (connects to Meta Simulator and starts polling)
adaptor.start()

# Subscribe to a topic
adaptor.subscribe("my/topic", lambda data, topic: print(f"Received: {data}"))

# Publish to a topic
adaptor.publish("another/topic", "Hello, world!")

# Stop the adaptor when done
adaptor.stop()
```

### Using the ROS Adaptor

```python
from common.ros_adaptor import ROSAdaptor
import rclpy

# Initialize ROS (if not already done)
rclpy.init()

# Create a ROS adaptor
adaptor = ROSAdaptor(
    node_name="my_ros_node",
    adaptor_id="my_ros_adaptor",
    topic_prefix="/my/topics/"  # Only handle topics with this prefix
)

# Start the adaptor
adaptor.start()

# Spin the ROS node
try:
    rclpy.spin(adaptor)
except KeyboardInterrupt:
    pass
finally:
    # Clean up
    adaptor.destroy_node()
    rclpy.shutdown()
```

### Using the Standalone Adaptor for Testing

```bash
# Run the standalone adaptor with an interactive shell
python -m common.standalone_adaptor --adaptor-id=test_adaptor --meta-simulator-url=http://localhost:5000
```

## Running the Pub/Sub Test

The test script demonstrates pub/sub functionality through the Meta Simulator:

```bash
# Start Meta Simulator separately
python -m meta_simulator.meta_simulator

# In another terminal, run the test
python -m tests.test_pub_sub --meta-simulator-url=http://localhost:5000

# Or let the test script start the Meta Simulator
python -m tests.test_pub_sub --start-simulator
```

## Extending the Framework

To create a custom adaptor:

1. Inherit from `AdaptorBase`
2. Override methods as needed
3. Add domain-specific functionality

Example:

```python
from common.adaptor_base import AdaptorBase

class MyCustomAdaptor(AdaptorBase):
    def __init__(self, adaptor_id=None, meta_simulator_url=None):
        super().__init__(adaptor_id, meta_simulator_url)
        # Add custom initialization
        
    def custom_method(self):
        # Add custom functionality
        pass
        
    def _process_message(self, message):
        # Override message processing if needed
        super()._process_message(message)
        # Add custom message processing
``` 