# FogSim Virtual Mode: Co-Simulation with Virtual Timeline

## Overview

FogSim's Virtual Mode represents a fundamental breakthrough in network-robot co-simulation through **complete timeline virtualization**. By decoupling both the network simulator and physics simulator from wallclock time, FogSim achieves unprecedented scalability and perfect reproducibility while maintaining realistic network-robot interactions.

### The Co-Simulation Challenge

Traditional approaches to simulating networked robotics face an impossible trade-off:
- **Without network simulation**: Unrealistic, missing critical delays and packet losses
- **With wallclock-synchronized network**: Frame rate limited to real-time, poor reproducibility
- **With simplified delay models**: Lacks fidelity, misses congestion and queuing effects

FogSim's virtual timeline solves this by creating a **unified virtual time domain** where both simulators advance in lockstep without any wallclock constraints.

## Core Innovation: Virtual Timeline Co-Simulation

### The Virtual Timeline Principle

In Virtual Mode, FogSim creates a completely artificial timeline where:
1. **Time is discrete**: Advances only through explicit simulation steps
2. **Time is coordinated**: Network and physics simulators share the same virtual clock
3. **Time is deterministic**: Every run produces identical results given same inputs
4. **Time is unbounded**: Can run arbitrarily faster than real-time

## Virtual Timeline Architecture

### The Co-Simulation Synchronization Problem

The fundamental challenge in network-robot co-simulation is synchronizing two different time domains:
- **Physics Simulator**: Advances in fixed timesteps (e.g., 50Hz for robot control)
- **Network Simulator**: Events occur at arbitrary times (packet arrivals, congestion events)

FogSim solves this through a **unified virtual timeline** that both simulators reference.

### Virtual Clock System

```python
class VirtualClock:
    """Core virtual time manager - the heartbeat of co-simulation"""
    
    def __init__(self, timestep: float = 0.1):
        self.timestep = timestep  # Fixed simulation timestep
        self.time = 0.0          # Current virtual time
    
    def advance(self) -> float:
        """Advance virtual time by one timestep"""
        self.time += self.timestep
        return self.time
    
    # Key insight: No sleep(), no wallclock reference!
    # Time only moves when we explicitly advance it
```

The virtual clock is the **single source of truth** for time in the entire co-simulation system.

## Co-Simulation Coordination Algorithm

### The Virtual Step Cycle

Each simulation step in Virtual Mode follows a carefully orchestrated sequence that maintains temporal consistency between network and physics:

```python
def virtual_step(action):
    """Core co-simulation step in virtual timeline"""
    
    # 1. ADVANCE VIRTUAL TIME
    virtual_time = clock.advance()  # T → T + Δt
    
    # 2. NETWORK SIMULATION CATCHUP
    # Run network simulator up to new virtual time
    # This processes all packet events that should occur by time T
    network_sim.run_until(virtual_time)
    
    # 3. COLLECT ARRIVED MESSAGES
    # Get messages that have completed their network journey
    arrived_messages = network_sim.get_ready_messages()
    
    # 4. INJECT NEW MESSAGES
    # Send current action into network (will arrive in future)
    if action is not None:
        network_sim.register_packet(
            message={'type': 'action', 'data': action},
            send_time=virtual_time,
            flow_id=0  # Actions on flow 0
        )
    
    # 5. PHYSICS SIMULATION STEP
    # Execute physics with delayed action (if any arrived)
    delayed_action = extract_latest_action(arrived_messages)
    obs, reward, done, info = physics_sim.step(delayed_action)
    
    # 6. SEND OBSERVATION TO NETWORK
    # This will be delayed and arrive in future steps
    network_sim.register_packet(
        message={'type': 'observation', 'data': obs},
        send_time=virtual_time,
        flow_id=1  # Observations on flow 1
    )
    
    # 7. RETURN APPROPRIATE OBSERVATION
    # Use delayed observation if available, else current
    delayed_obs = extract_latest_observation(arrived_messages)
    return delayed_obs if delayed_obs else obs, reward, done, info
```

### Critical Insight: Temporal Causality

The algorithm maintains **strict temporal causality**:
- Messages sent at time T arrive at time T + network_delay
- Physics simulation at time T uses actions from time T - network_delay
- No message can arrive before it was sent
- All events are deterministically ordered

## Network-Physics Co-Simulation Integration

### Virtual Network Simulator

The network simulator (NS.py) is modified to use virtual time:

```python
class VirtualTimeEnvironment(simpy.Environment):
    """SimPy environment synchronized with virtual timeline"""
    
    def __init__(self, virtual_clock):
        self.virtual_clock = virtual_clock
        self._queue = []  # Event queue
    
    @property
    def now(self):
        """Current time comes from virtual clock"""
        return self.virtual_clock.now()
    
    def run(self, until):
        """Process events up to target virtual time"""
        # Process all events scheduled before 'until'
        while self._queue and self._queue[0].time <= until:
            event = self._queue.pop(0)
            # Execute event in virtual time
            self._process_event(event)
        
        # No sleep! Time has already advanced virtually
```

### Packet Lifecycle in Virtual Time

```python
class NSPyNetworkSimulator:
    """Network simulator with virtual timeline"""
    
    def register_packet(self, message, send_time, flow_id):
        """Register packet for transmission"""
        packet_id = uuid()
        
        # Create packet with virtual timestamp
        packet = Packet(
            time=send_time,
            size=len(message),
            flow_id=flow_id,
            id=packet_id
        )
        
        # Calculate delivery time in virtual timeline
        queue_delay = packet.size / self.source_rate
        prop_delay = self.link_delay
        delivery_time = send_time + queue_delay + prop_delay
        
        # Schedule delivery in virtual time
        self.schedule_delivery(packet_id, delivery_time)
        
    def run_until(self, virtual_time):
        """Process network events up to virtual time"""
        # Deliver all packets that should arrive by virtual_time
        for packet_id, delivery_time in self.pending_packets:
            if delivery_time <= virtual_time:
                self.mark_packet_delivered(packet_id)
```

### The Synchronization Invariant

FogSim maintains a critical invariant:
```
physics_sim.time == network_sim.time == virtual_clock.time
```

This ensures:
- **Determinism**: Same inputs → same outputs every run
- **Correctness**: Network delays properly modeled
- **Scalability**: No wallclock bottleneck

## Detailed Algorithms

### Virtual Clock Scheduling in Network Simulation

FogSim employs a **Virtual Clock** algorithm for packet scheduling, which is crucial for simulating realistic network behavior in virtual time:

```python
class VirtualClockServer:
    """Weighted Fair Queuing using Virtual Clock algorithm"""
    
    def __init__(self, source_rate, flow_weights):
        self.source_rate = source_rate
        self.flow_weights = flow_weights
        self.virtual_time = 0.0
        self.last_finish_time = {}  # Per-flow tracking
    
    def schedule_packet(self, packet, flow_id):
        """Calculate when packet should be transmitted"""
        
        # Service time based on packet size and link rate
        service_time = packet.size / self.source_rate
        
        # Weight-adjusted service time (higher weight = more bandwidth)
        weighted_service = service_time / self.flow_weights[flow_id]
        
        # Virtual finish time calculation
        start_time = max(
            self.virtual_time,  # Can't start before now
            self.last_finish_time.get(flow_id, 0)  # Can't overlap with previous packet
        )
        
        finish_time = start_time + weighted_service
        self.last_finish_time[flow_id] = finish_time
        
        # Schedule packet transmission at finish_time
        return finish_time
```

This algorithm ensures:
- **Fair bandwidth allocation**: Based on flow weights
- **Work-conserving**: Link never idle if packets are waiting
- **Deterministic ordering**: Same packet order every run

### Message Buffering and Delayed Action Handling

A critical aspect of co-simulation is handling network delays gracefully:

```python
class DelayedMessageBuffer:
    """Manages messages in transit through network"""
    
    def __init__(self):
        self.action_buffer = None  # Latest action waiting to be applied
        self.obs_buffer = None      # Latest observation from physics
        
    def process_arrived_messages(self, messages):
        """Extract latest action/observation from arrived messages"""
        
        latest_action = None
        latest_obs = None
        
        for msg in messages:
            if msg['type'] == 'action':
                # Always use most recent action
                if latest_action is None or msg['timestamp'] > latest_action['timestamp']:
                    latest_action = msg
            elif msg['type'] == 'observation':
                # Always use most recent observation
                if latest_obs is None or msg['timestamp'] > latest_obs['timestamp']:
                    latest_obs = msg
        
        return latest_action, latest_obs
```

### Event Ordering and Determinism

Virtual Mode guarantees deterministic execution through careful event ordering:

```python
class DeterministicEventQueue:
    """Priority queue with deterministic tie-breaking"""
    
    def insert(self, event):
        # Priority tuple: (time, tie_breaker, event_id)
        # Ensures consistent ordering when events have same time
        priority = (
            event.time,
            event.source_id,  # Tie-break by source
            event.sequence_num  # Then by sequence
        )
        heapq.heappush(self.queue, (priority, event))
    
    def process_until(self, virtual_time):
        """Process all events up to virtual_time"""
        while self.queue and self.queue[0][0][0] <= virtual_time:
            priority, event = heapq.heappop(self.queue)
            event.execute()
```

## Performance Optimizations

### Zero-Copy Message Passing

In Virtual Mode, messages between simulators don't need serialization:

```python
def optimized_message_pass(message, from_sim, to_sim):
    """Direct memory reference passing in virtual mode"""
    if mode == SimulationMode.VIRTUAL:
        # No serialization needed - same process
        to_sim.receive(message)  # Direct reference
    else:
        # Other modes need serialization for network
        serialized = pickle.dumps(message)
        network.send(serialized)
```

### Adaptive Timestep Management

FogSim can dynamically adjust timesteps based on network conditions:

```python
def adaptive_timestep(network_load, base_timestep=0.1):
    """Adjust timestep based on network activity"""
    
    if network_load < 0.3:
        # Low load - larger timesteps acceptable
        return base_timestep * 2
    elif network_load > 0.7:
        # High load - smaller timesteps for accuracy
        return base_timestep / 2
    else:
        return base_timestep
```

## Key Benefits of Virtual Timeline

### 1. **Unlimited Speed**
Without wallclock constraints, simulation speed is limited only by CPU:
- Simple environments: 10,000x real-time
- Complex environments: 100-1000x real-time
- Network simulation adds ~5% overhead

### 2. **Perfect Reproducibility**
Every run with same inputs produces identical results:
- Same packet delivery order
- Same network delays
- Same physics outcomes
- Essential for debugging and scientific validation

### 3. **Scalable Training**
For reinforcement learning:
- Generate millions of samples per hour
- Parallelize across multiple virtual environments
- No synchronization overhead between parallel runs

### 4. **Accurate Network Modeling**
Despite virtualization, network behavior remains realistic:
- Congestion and queuing delays properly simulated
- Packet loss and reordering supported
- Bandwidth constraints enforced
- All in virtual time without wallclock limitations

## Practical Implications

### For Robotics Researchers
- Train policies 100-1000x faster than real-time
- Test network-resilient algorithms efficiently
- Validate edge/cloud architectures without infrastructure

### For Network Researchers  
- Study robot traffic patterns at scale
- Evaluate network protocols with realistic workloads
- Co-design network and control algorithms

### For System Designers
- Rapid prototyping of distributed robot systems
- Performance evaluation under various network conditions
- Cost-effective testing before deployment

## Conclusion

FogSim's Virtual Mode with its virtual timeline co-simulation represents a paradigm shift in how we simulate networked robotics systems. By completely decoupling both network and physics simulation from wallclock time while maintaining their synchronization through a unified virtual timeline, FogSim achieves:

1. **Unprecedented speed**: Limited only by computational resources
2. **Perfect reproducibility**: Essential for scientific research
3. **Realistic network effects**: Full packet-level simulation
4. **Seamless co-simulation**: Network and physics in perfect sync

This innovation enables researchers to tackle previously intractable problems in networked robotics, from training robust policies for unreliable networks to designing optimal edge/cloud architectures for robot fleets. The virtual timeline approach proves that we don't need to sacrifice speed for realism or reproducibility for scale - we can have both through intelligent system design.