# CloudSim with gRPC

This project implements a cloud-based simulator architecture using gRPC for efficient communication between components.

## Key Components

### Meta Simulator
The central coordination system that maintains simulation time and message passing between adaptors.

### Environment Adaptor
Connects to simulation environments and forwards their state to the Meta Simulator.

### Algorithm Adaptor
Connects to algorithm implementations and forwards their commands/responses to the Meta Simulator.

## Communication Architecture

The system uses gRPC for all component communication:

1. **Streamlined Polling**: Both environment and algorithm adaptors poll the Meta Simulator for messages, which eliminates the need for separate register and send operations.

2. **Bidirectional Communication**: Messages flow naturally through the polling mechanism:
   - Environment adaptor publishes environment states that are returned in algorithm adaptor's poll responses
   - Algorithm adaptor publishes algorithm responses that are returned in environment adaptor's poll responses

3. **Protocol Buffer Messages**: All messages are defined using Protocol Buffers in `messages.proto`.

## Running the System

1. Start the Meta Simulator:
   ```
   cd cloudsim
   python -m cloudsim.meta_simulator.meta_simulator
   ```

2. Start the Environment Adaptor:
   ```
   cd cloudsim
   python -m cloudsim.simulator_adaptor.simulator_adaptor
   ```

3. Start the Algorithm Adaptor:
   ```
   cd cloudsim
   python -m cloudsim.algorithm_adaptor.algorithm_adaptor
   ```

## Configuration

Each component can be configured using environment variables:

- `META_SIMULATOR_URL`: gRPC server address (default: `localhost:50051`)
- `ADAPTOR_ID`: Unique ID for each adaptor
- `SIMULATOR_TOPIC_PREFIX`/`ALGORITHM_TOPIC_PREFIX`: Prefix for topic discovery
- `POLL_INTERVAL`: How often to poll for updates (in seconds)
- `TOPIC_DISCOVERY_INTERVAL`: How often to discover new topics (in seconds)
- `GRPC_PORT`: Port for the Meta Simulator gRPC server (default: `50051`)

## Benefits of the New Architecture

1. **Simplified Communication**: No need for separate register and send endpoints
2. **Reduced Overhead**: gRPC is more efficient than HTTP+JSON for internal communication
3. **Type Safety**: Protocol Buffers provide strong typing for all messages
4. **Streamlined Code**: Cleaner implementation with fewer edge cases
5. **Better Performance**: Lower latency and higher throughput 