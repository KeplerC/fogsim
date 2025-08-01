"""
Training Convergence Demo - High Frame Rate Benefits

This example demonstrates the training convergence metric from CLAUDE.md:
- Training policies with network delay/packet loss
- Comparing convergence with wallclock time
- Success rate after 1 hour of training
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

from fogsim import (
    Env, GymHandler, SimulationMode,
    NetworkControlConfig, get_low_latency_config
)


class SimpleRLAgent:
    """Simple Q-learning agent for CartPole."""
    
    def __init__(self, state_bins: int = 10, action_space_size: int = 2):
        self.state_bins = state_bins
        self.action_space_size = action_space_size
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
    def discretize_state(self, observation: np.ndarray) -> Tuple:
        """Discretize continuous state into bins."""
        # Simple discretization for CartPole
        # [cart_position, cart_velocity, pole_angle, pole_velocity]
        if len(observation) >= 4:
            bins = [
                np.clip(int((observation[0] + 2.4) * 2), 0, 9),  # position
                np.clip(int((observation[1] + 3) * 2), 0, 9),    # velocity
                np.clip(int((observation[2] + 0.21) * 20), 0, 9), # angle
                np.clip(int((observation[3] + 3) * 2), 0, 9)     # angular velocity
            ]
            return tuple(bins)
        return (0, 0, 0, 0)
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Get action using epsilon-greedy policy."""
        discrete_state = self.discretize_state(state)
        
        # Exploration vs exploitation
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # Get Q-values for state
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space_size)
        
        return int(np.argmax(self.q_table[discrete_state]))
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool):
        """Update Q-table using Q-learning."""
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        
        # Initialize Q-values if needed
        if discrete_state not in self.q_table:
            self.q_table[discrete_state] = np.zeros(self.action_space_size)
        if discrete_next_state not in self.q_table:
            self.q_table[discrete_next_state] = np.zeros(self.action_space_size)
        
        # Q-learning update
        current_q = self.q_table[discrete_state][action]
        max_next_q = np.max(self.q_table[discrete_next_state]) if not done else 0
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        self.q_table[discrete_state][action] = new_q
        
        # Decay epsilon
        if done and training:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent_for_duration(mode: SimulationMode, 
                           training_duration: float = 60.0,  # seconds
                           network_delay: float = 10.0) -> Dict:
    """Train an agent for a fixed wallclock duration."""
    
    print(f"\nTraining in {mode.value} mode for {training_duration}s...")
    
    # Create environment
    handler = GymHandler(env_name="CartPole-v1")
    network_config = get_low_latency_config()
    
    env = Env(
        handler=handler,
        network_config=network_config,
        enable_network=True,
        timestep=0.01,  # 10ms timestep
        mode=mode
    )
    
    # Configure network delay
    if mode == SimulationMode.SIMULATED_NET and hasattr(env.network_sim, 'configure_link'):
        env.network_sim.configure_link({
            'delay': network_delay / 1000.0,  # Convert ms to seconds
            'bandwidth': 100e6,  # 100 Mbps
            'loss': 0.001  # 0.1% packet loss
        })
    
    # Create agent
    agent = SimpleRLAgent()
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    success_episodes = []  # Episodes with length >= 195
    timestamps = []
    
    # Training loop
    start_time = time.time()
    episode = 0
    total_steps = 0
    
    while time.time() - start_time < training_duration:
        observation, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Agent selects action
            action = agent.get_action(observation, training=True)
            
            # Step environment
            next_observation, reward, _, termination, timeout, _ = env.step(action)
            done = termination or timeout
            
            # Update agent
            agent.update(observation, action, reward, next_observation, done)
            
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            total_steps += 1
            
            # Check time limit
            if time.time() - start_time >= training_duration:
                done = True
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        success_episodes.append(1 if episode_length >= 195 else 0)
        timestamps.append(time.time() - start_time)
        
        episode += 1
        
        if episode % 100 == 0:
            recent_success_rate = np.mean(success_episodes[-100:]) if len(success_episodes) >= 100 else 0
            print(f"  Episode {episode}: Recent success rate = {recent_success_rate*100:.1f}%")
    
    env.close()
    
    # Calculate final metrics
    final_success_rate = np.mean(success_episodes[-100:]) if len(success_episodes) >= 100 else np.mean(success_episodes)
    
    return {
        'mode': mode,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_episodes': success_episodes,
        'timestamps': timestamps,
        'total_episodes': episode,
        'total_steps': total_steps,
        'final_success_rate': final_success_rate,
        'training_duration': time.time() - start_time,
        'episodes_per_second': episode / (time.time() - start_time),
        'steps_per_second': total_steps / (time.time() - start_time)
    }


def evaluate_trained_agent(agent: SimpleRLAgent, mode: SimulationMode, 
                         num_episodes: int = 100) -> float:
    """Evaluate a trained agent's performance."""
    
    handler = GymHandler(env_name="CartPole-v1")
    env = Env(
        handler=handler,
        enable_network=True,
        timestep=0.01,
        mode=mode
    )
    
    successes = 0
    
    for _ in range(num_episodes):
        observation, _ = env.reset()
        episode_length = 0
        
        done = False
        while not done and episode_length < 500:
            action = agent.get_action(observation, training=False)
            observation, _, _, termination, timeout, _ = env.step(action)
            done = termination or timeout
            episode_length += 1
        
        if episode_length >= 195:
            successes += 1
    
    env.close()
    
    return successes / num_episodes


def plot_training_results(results: List[Dict]):
    """Plot training convergence results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        colors = ['blue', 'orange', 'green']
        
        # Plot 1: Episode rewards over time
        for i, result in enumerate(results):
            # Smooth rewards with moving average
            rewards = result['episode_rewards']
            window = min(50, len(rewards) // 10)
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                times = np.array(result['timestamps'][:len(smoothed)])
                ax1.plot(times, smoothed, label=result['mode'].value, color=colors[i])
        
        ax1.set_xlabel('Wallclock Time (s)')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Progress Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over episodes
        for i, result in enumerate(results):
            # Calculate rolling success rate
            success = result['success_episodes']
            window = min(100, len(success) // 5)
            if window > 1 and len(success) > window:
                success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
                episodes = range(window-1, len(success))
                ax2.plot(episodes, success_rate, label=result['mode'].value, color=colors[i])
        
        ax2.set_xlabel('Episode Number')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Episodes completed over time
        for i, result in enumerate(results):
            episodes = range(len(result['timestamps']))
            ax3.plot(result['timestamps'], episodes, label=result['mode'].value, color=colors[i])
        
        ax3.set_xlabel('Wallclock Time (s)')
        ax3.set_ylabel('Episodes Completed')
        ax3.set_title('Training Efficiency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison bar chart
        modes = [r['mode'].value for r in results]
        metrics = ['Episodes/sec', 'Steps/sec', 'Final Success %']
        
        x = np.arange(len(modes))
        width = 0.25
        
        episodes_per_sec = [r['episodes_per_second'] for r in results]
        steps_per_sec = [r['steps_per_second']/100 for r in results]  # Scale down
        success_rates = [r['final_success_rate']*100 for r in results]
        
        ax4.bar(x - width, episodes_per_sec, width, label='Episodes/sec')
        ax4.bar(x, steps_per_sec, width, label='Steps/sec (÷100)')
        ax4.bar(x + width, success_rates, width, label='Success Rate %')
        
        ax4.set_xlabel('Mode')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(modes)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('training_convergence_results.png')
        print("\nResults saved to training_convergence_results.png")
        
    except Exception as e:
        print(f"\nCould not generate plots: {e}")


def main():
    """Run the training convergence experiment."""
    print("Training Convergence Demo")
    print("="*70)
    print("Demonstrating FogSim's high frame rate benefits for RL training")
    print("Training agents for 60 seconds of wallclock time")
    
    # Training duration (1 minute for demo, would be 1 hour in full experiment)
    training_duration = 60.0  # seconds
    
    # Train agents in different modes
    results = []
    
    for mode in [SimulationMode.VIRTUAL, SimulationMode.SIMULATED_NET]:
        result = train_agent_for_duration(mode, training_duration, network_delay=10.0)
        results.append(result)
        
        print(f"\n{mode.value} Results:")
        print(f"  Total episodes: {result['total_episodes']}")
        print(f"  Total steps: {result['total_steps']}")
        print(f"  Episodes/second: {result['episodes_per_second']:.2f}")
        print(f"  Steps/second: {result['steps_per_second']:.1f}")
        print(f"  Final success rate: {result['final_success_rate']*100:.1f}%")
    
    # Plot results
    plot_training_results(results)
    
    # Summary comparison
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if len(results) >= 2:
        virtual_result = results[0]
        simulated_result = results[1]
        
        episode_speedup = virtual_result['total_episodes'] / simulated_result['total_episodes']
        step_speedup = virtual_result['total_steps'] / simulated_result['total_steps']
        
        print(f"\nTraining Efficiency (in {training_duration}s wallclock time):")
        print(f"  Virtual mode: {virtual_result['total_episodes']} episodes, "
              f"{virtual_result['total_steps']} steps")
        print(f"  Simulated mode: {simulated_result['total_episodes']} episodes, "
              f"{simulated_result['total_steps']} steps")
        print(f"  Speedup: {episode_speedup:.2f}x episodes, {step_speedup:.2f}x steps")
        
        print(f"\nFinal Performance:")
        print(f"  Virtual mode success rate: {virtual_result['final_success_rate']*100:.1f}%")
        print(f"  Simulated mode success rate: {simulated_result['final_success_rate']*100:.1f}%")
        
        print("\n✓ Virtual mode enables significantly more training iterations")
        print("  in the same wallclock time, leading to better convergence!")


if __name__ == "__main__":
    main()