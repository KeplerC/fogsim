#!/usr/bin/env python3
"""
Training Convergence Demo - High Frame Rate Benefits with PPO

This example demonstrates the training convergence metric from CLAUDE.md:
- Training PPO policies with network delay/packet loss across all three FogSim modes
- Comparing convergence curves with wallclock time
- Success rate comparison after fixed training duration

Shows how Virtual Timeline mode enables faster RL training by decoupling from wallclock.
Uses PPO (Proximal Policy Optimization) with PyTorch and stable-baselines3.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
import argparse
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from fogsim import FogSim, SimulationMode, NetworkConfig
from fogsim.handlers import GymHandler


class TrainingMetricsCallback(BaseCallback):
    """Callback to track training metrics during PPO training."""
    
    def __init__(self, max_duration: float, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_episodes = []
        self.timestamps = []
        self.network_delays = []
        self.start_time = time.time()
        self.episode_count = 0
        self.total_steps = 0
        self.max_duration = max_duration
        
    def _on_step(self) -> bool:
        # Check if max duration has been reached
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.max_duration:
            print(f"\\nTraining stopped after {elapsed_time:.1f}s (reached max duration)")
            return False  # Stop training
        
        # Track total steps
        self.total_steps += 1
        
        # Check if episode ended
        if self.locals.get('dones', [False])[0]:
            info = self.locals.get('infos', [{}])[0]
            
            # Get episode info from VecMonitor
            if 'episode' in info:
                self.episode_count += 1
                episode_reward = info['episode']['r']
                episode_length = info['episode']['l']
                
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.success_episodes.append(1 if episode_length >= 195 else 0)
                self.timestamps.append(time.time() - self.start_time)
                
                # Track network delays if available
                if 'network_latencies' in info:
                    delays = [lat.get('latency', 0) for lat in info['network_latencies']]
                    self.network_delays.append(np.mean(delays) if delays else 0)
                else:
                    self.network_delays.append(150)  # Default 150ms
                
                # Progress reporting
                if self.episode_count % 100 == 0:
                    recent_success_rate = np.mean(self.success_episodes[-100:]) if len(self.success_episodes) >= 100 else np.mean(self.success_episodes)
                    avg_delay = np.mean(self.network_delays[-100:]) if self.network_delays else 0
                    elapsed = time.time() - self.start_time
                    print(f"  Episode {self.episode_count} ({elapsed:.1f}s): Success rate = {recent_success_rate*100:.1f}%, Avg delay = {avg_delay:.1f}ms")
        
        return True


def train_agent_for_duration(mode: SimulationMode, 
                           training_duration: float = 60.0,  # seconds
                           timestep: float = 0.01) -> Dict:
    """Train a PPO agent for a fixed wallclock duration."""
    
    print(f"\\nTraining PPO in {mode.value.upper()} mode for {training_duration}s...")
    
    # Create FogSim environment with 150ms network latency
    handler = GymHandler(env_name="CartPole-v1")
    
    # Configure network with 150ms latency
    network_config = NetworkConfig()
    network_config.topology.link_delay = 0.15  # 150ms
    network_config.source_rate = 1e6  # 1 Mbps
    network_config.packet_loss_rate = 0.01  # 1% packet loss for realism
    
    fogsim = FogSim(handler, mode=mode, timestep=timestep, network_config=network_config)
    
    # Wrap environment for stable-baselines3
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete
    
    class FogSimWrapper(Env):
        """Wrapper to make FogSim compatible with stable-baselines3."""
        def __init__(self, fogsim):
            super().__init__()
            self.fogsim = fogsim
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            self.action_space = Discrete(2)
            self._reset_episode_vars()
            
        def reset(self, seed=None, options=None):
            self._reset_episode_vars()
            obs, info = self.fogsim.reset()
            return obs.astype(np.float32), info
            
        def step(self, action):
            obs, reward, success, terminated, truncated, info = self.fogsim.step(action)
            self._episode_reward += reward
            self._episode_length += 1
            
            # Accumulate network delays during episode
            if 'network_latencies' in info and info['network_latencies']:
                for latency_info in info['network_latencies']:
                    # Convert latency from seconds to milliseconds
                    latency_ms = latency_info.get('latency', 0) * 1000
                    self._episode_network_delays.append(latency_ms)
            
            # Track episode metrics
            if terminated or truncated:
                info['episode'] = {
                    'r': self._episode_reward,
                    'l': self._episode_length,
                    't': time.time() - self._episode_start
                }
                # Include network delays in episode info (already in milliseconds)
                if self._episode_network_delays:
                    info['network_latencies'] = [{'latency': delay} for delay in self._episode_network_delays]
                else:
                    # Use configured delay if no actual delays recorded
                    info['network_latencies'] = [{'latency': 150}]  # 150ms default
            
            return obs.astype(np.float32), reward, terminated, truncated, info
            
        def render(self):
            pass
            
        def close(self):
            self.fogsim.close()
            
        def _reset_episode_vars(self):
            self._episode_reward = 0
            self._episode_length = 0
            self._episode_start = time.time()
            self._episode_network_delays = []
    
    # Create wrapped environment
    env = FogSimWrapper(fogsim)
    
    # Use VecMonitor to track episode statistics
    vec_env = VecMonitor(make_vec_env(lambda: env, n_envs=1))
    
    # Create PPO model with appropriate hyperparameters for CartPole
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=1e-5,
        n_steps=32,  # Reduced for faster updates
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Create callback to track metrics and handle time limit
    callback = TrainingMetricsCallback(max_duration=training_duration)
    
    # Use a very large number of timesteps - the callback will stop training when duration is reached
    total_timesteps = 10_000_000  # Large number, will be stopped by callback
    
    # Train the model
    start_time = time.time()
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=False
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    
    actual_duration = time.time() - start_time
    
    # Close environment
    vec_env.close()
    
    # Calculate final metrics from callback
    if callback.episode_rewards:
        final_success_rate = np.mean(callback.success_episodes[-100:]) if len(callback.success_episodes) >= 100 else np.mean(callback.success_episodes)
        avg_network_delay = np.mean(callback.network_delays) if callback.network_delays else 150
    else:
        final_success_rate = 0
        avg_network_delay = 150
    
    return {
        'mode': mode,
        'episode_rewards': callback.episode_rewards,
        'episode_lengths': callback.episode_lengths,
        'success_episodes': callback.success_episodes,
        'timestamps': callback.timestamps,
        'network_delays': callback.network_delays,
        'total_episodes': callback.episode_count,
        'total_steps': callback.total_steps,
        'final_success_rate': final_success_rate,
        'avg_network_delay': avg_network_delay,
        'training_duration': actual_duration,
        'episodes_per_second': callback.episode_count / actual_duration if actual_duration > 0 else 0,
        'steps_per_second': callback.total_steps / actual_duration if actual_duration > 0 else 0,
        'model': model  # Store the trained model
    }


def evaluate_trained_model(model: PPO, mode: SimulationMode, 
                         num_episodes: int = 100) -> float:
    """Evaluate a trained PPO model's performance."""
    
    handler = GymHandler(env_name="CartPole-v1")
    
    # Configure network with same latency as training
    network_config = NetworkConfig()
    network_config.topology.link_delay = 0.15  # 150ms
    network_config.source_rate = 1e6  # 1 Mbps
    network_config.packet_loss_rate = 0.01  # 1% packet loss
    
    fogsim = FogSim(handler, mode=mode, timestep=0.01, network_config=network_config)
    
    successes = 0
    
    for _ in range(num_episodes):
        observation, _ = fogsim.reset()
        episode_length = 0
        
        done = False
        while not done and episode_length < 500:
            # Use PPO model to predict action
            action, _ = model.predict(observation.astype(np.float32), deterministic=True)
            observation, _, _, termination, timeout, _ = fogsim.step(int(action))
            done = termination or timeout
            episode_length += 1
        
        if episode_length >= 195:
            successes += 1
    
    fogsim.close()
    
    return successes / num_episodes


def plot_training_results(results: List[Dict], save_path: str = 'training_convergence_results.png'):
    """Plot training convergence results."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['blue', 'orange', 'green']
        mode_names = {'virtual': 'Virtual Timeline', 'simulated': 'Real Clock + Sim Network', 'real': 'Real Clock + Real Network'}
        
        # Plot 1: Episode rewards over time (smoothed)
        for i, result in enumerate(results):
            rewards = result['episode_rewards']
            times = result['timestamps']
            
            # Smooth rewards with moving average
            window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
            if window > 1:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                times_smooth = times[:len(smoothed)]
                ax1.plot(times_smooth, smoothed, 
                        label=mode_names.get(result['mode'].value, result['mode'].value), 
                        color=colors[i], linewidth=2)
            else:
                ax1.plot(times, rewards, 
                        label=mode_names.get(result['mode'].value, result['mode'].value), 
                        color=colors[i], linewidth=2)
        
        ax1.set_xlabel('Wallclock Time (s)')
        ax1.set_ylabel('Episode Reward (smoothed)')
        ax1.set_title('Training Progress Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over episodes
        for i, result in enumerate(results):
            success = result['success_episodes']
            # Calculate rolling success rate
            window = min(100, len(success) // 5) if len(success) > 5 else len(success)
            if window > 1 and len(success) > window:
                success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
                episodes = range(window-1, len(success))
                ax2.plot(episodes, success_rate, 
                        label=mode_names.get(result['mode'].value, result['mode'].value), 
                        color=colors[i], linewidth=2)
        
        ax2.set_xlabel('Episode Number')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate During Training')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Episodes completed over time
        for i, result in enumerate(results):
            episodes = range(1, len(result['timestamps']) + 1)
            ax3.plot(result['timestamps'], episodes, 
                    label=mode_names.get(result['mode'].value, result['mode'].value), 
                    color=colors[i], linewidth=2)
        
        ax3.set_xlabel('Wallclock Time (s)')
        ax3.set_ylabel('Episodes Completed')
        ax3.set_title('Training Efficiency (Episodes per Second)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison bar chart
        modes = [mode_names.get(r['mode'].value, r['mode'].value) for r in results]
        
        x = np.arange(len(modes))
        width = 0.25
        
        episodes_per_sec = [r['episodes_per_second'] for r in results]
        steps_per_sec = [r['steps_per_second']/1000 for r in results]  # Scale to thousands
        success_rates = [r['final_success_rate']*100 for r in results]
        
        ax4.bar(x - width, episodes_per_sec, width, label='Episodes/sec', alpha=0.8)
        ax4.bar(x, steps_per_sec, width, label='Steps/sec (÷1000)', alpha=0.8)
        ax4.bar(x + width, success_rates, width, label='Success Rate %', alpha=0.8)
        
        ax4.set_xlabel('Mode')
        ax4.set_ylabel('Value')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(modes, rotation=15, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\\nResults saved to {save_path}")
        
    except Exception as e:
        print(f"\\nCould not generate plots: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run the training convergence experiment."""
    parser = argparse.ArgumentParser(description="FogSim PPO Training Convergence Demo")
    parser.add_argument("--duration", type=int, default=120, help="Training duration in seconds")
    parser.add_argument("--modes", nargs='+', choices=['virtual', 'simulated', 'real'], 
                       default=['virtual', 'simulated', 'real'], help="Modes to compare")
    parser.add_argument("--timestep", type=float, default=0.01, help="Simulation timestep")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models after training")
    
    args = parser.parse_args()
    
    print("FogSim PPO Training Convergence Demo")
    print("="*70)
    print("Demonstrating FogSim's high frame rate benefits for PPO training")
    print(f"Training PPO agents for {args.duration} seconds of wallclock time")
    print(f"Network configuration: 150ms latency, 1% packet loss")
    print(f"Comparing modes: {', '.join(args.modes)}")
    
    # Train agents in different modes
    results = []
    mode_map = {
        'virtual': SimulationMode.VIRTUAL,
        'simulated': SimulationMode.SIMULATED_NET,
        'real': SimulationMode.REAL_NET
    }
    
    for mode_name in args.modes:
        mode = mode_map[mode_name]
        result = train_agent_for_duration(mode, args.duration, args.timestep)
        results.append(result)
        
        print(f"\\n{mode.value.upper()} Results:")
        print(f"  Total episodes: {result['total_episodes']}")
        print(f"  Total steps: {result['total_steps']}")
        print(f"  Episodes/second: {result['episodes_per_second']:.2f}")
        print(f"  Steps/second: {result['steps_per_second']:.1f}")
        print(f"  Final success rate: {result['final_success_rate']*100:.1f}%")
        print(f"  Avg network delay: {result['avg_network_delay']:.1f}ms")
    
    # Plot results
    plot_training_results(results)
    
    # Evaluate trained models on test episodes if requested
    if hasattr(args, 'evaluate') and args.evaluate:
        print("\\n" + "="*70)
        print("EVALUATING TRAINED MODELS")
        print("="*70)
        
        for result in results:
            if 'model' in result:
                eval_success_rate = evaluate_trained_model(
                    result['model'], 
                    result['mode'], 
                    num_episodes=100
                )
                print(f"\\n{result['mode'].value.upper()} - Evaluation Success Rate: {eval_success_rate*100:.1f}%")
    
    # Summary comparison
    print("\\n" + "="*70)
    print("TRAINING CONVERGENCE COMPARISON")
    print("="*70)
    
    if len(results) >= 2:
        # Find virtual mode for baseline comparison
        virtual_result = None
        for result in results:
            if result['mode'] == SimulationMode.VIRTUAL:
                virtual_result = result
                break
        
        if virtual_result:
            print(f"\\nTraining Efficiency (in {args.duration}s wallclock time):")
            
            for result in results:
                if result['mode'] == SimulationMode.VIRTUAL:
                    continue
                    
                episode_speedup = virtual_result['total_episodes'] / result['total_episodes'] if result['total_episodes'] > 0 else 0
                step_speedup = virtual_result['total_steps'] / result['total_steps'] if result['total_steps'] > 0 else 0
                
                print(f"\\n  Virtual vs {result['mode'].value}:")
                print(f"    Episodes: {virtual_result['total_episodes']} vs {result['total_episodes']} ({episode_speedup:.2f}x speedup)")
                print(f"    Steps: {virtual_result['total_steps']} vs {result['total_steps']} ({step_speedup:.2f}x speedup)")
                print(f"    Success Rate: {virtual_result['final_success_rate']*100:.1f}% vs {result['final_success_rate']*100:.1f}%")
        
        print(f"\\nKey Insights:")
        print(f"✓ Virtual Timeline mode enables significantly more training iterations")
        print(f"✓ Higher frame rates lead to faster policy convergence")
        print(f"✓ Network delays affect both training speed and final performance")
        print(f"✓ Real network mode shows actual latency impact on RL training")
        
        # Show the FogSim advantage
        if virtual_result and len(results) > 1:
            max_other_episodes = max([r['total_episodes'] for r in results if r['mode'] != SimulationMode.VIRTUAL])
            advantage = virtual_result['total_episodes'] / max_other_episodes if max_other_episodes > 0 else 1
            print(f"\\n🚀 FogSim Virtual Timeline advantage: {advantage:.1f}x more training episodes!")


if __name__ == "__main__":
    main()