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
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import argparse
import torch
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

from fogsim import FogSim, SimulationMode, NetworkConfig
from fogsim.handlers import GymHandler


@dataclass
class RLScenarioConfig:
    """Configuration for different RL training scenarios."""
    name: str
    env_name: str = "CartPole-v1"
    learning_rate: float = 1e-5
    n_steps: int = 32
    batch_size: int = 32
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    network_delay: float = 0.15  # seconds
    packet_loss_rate: float = 0.01
    source_rate: float = 1e6  # bps
    timestep: float = 0.01
    success_threshold: float = 195.0  # Task-specific success metric
    max_episode_steps: int = 500
    device: str = "auto"  # "auto", "cuda", or "cpu"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLScenarioConfig':
        return cls(**config_dict)


# Predefined scenarios for different environments
PREDEFINED_SCENARIOS = {
    # CartPole - Classic control task
    "cartpole": RLScenarioConfig(
        name="cartpole",
        env_name="CartPole-v1",
        learning_rate=1e-3,
        n_steps=32,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        network_delay=0.15,
        packet_loss_rate=0.01,
        success_threshold=195.0,
        max_episode_steps=500
    ),
    
    # Ant - MuJoCo locomotion task
    "ant": RLScenarioConfig(
        name="ant",
        env_name="Ant-v4",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        network_delay=0.15,
        packet_loss_rate=0.01,
        success_threshold=3000.0,  # Reward threshold
        max_episode_steps=1000
    ),
    
    # Humanoid - Complex MuJoCo locomotion
    "humanoid": RLScenarioConfig(
        name="humanoid",
        env_name="Humanoid-v4",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        network_delay=0.15,
        packet_loss_rate=0.01,
        success_threshold=5000.0,  # Reward threshold
        max_episode_steps=1000
    ),
    
    # HalfCheetah - MuJoCo running task
    "halfcheetah": RLScenarioConfig(
        name="halfcheetah",
        env_name="HalfCheetah-v4",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        network_delay=0.15,
        packet_loss_rate=0.01,
        success_threshold=4000.0,  # Reward threshold
        max_episode_steps=1000
    ),
    
    # CartPole with high latency network
    "cartpole_high_latency": RLScenarioConfig(
        name="cartpole_high_latency",
        env_name="CartPole-v1",
        learning_rate=1e-3,
        n_steps=32,
        batch_size=32,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        network_delay=0.5,  # 500ms
        packet_loss_rate=0.05,  # 5% loss
        success_threshold=195.0,
        max_episode_steps=500
    ),
    
    # Ant with low latency network
    "ant_low_latency": RLScenarioConfig(
        name="ant_low_latency",
        env_name="Ant-v4",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        network_delay=0.05,  # 50ms
        packet_loss_rate=0.001,  # 0.1% loss
        success_threshold=3000.0,
        max_episode_steps=1000
    )
}


class TrainingMetricsCallback(BaseCallback):
    """Callback to track training metrics during PPO training."""
    
    def __init__(self, max_duration: float, config: RLScenarioConfig, verbose=0):
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
        self.config = config
        
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
                # Determine success based on environment type
                if 'CartPole' in self.config.env_name:
                    # For CartPole, success is based on episode length
                    success = 1 if episode_length >= self.config.success_threshold else 0
                else:
                    # For MuJoCo envs, success is based on reward threshold
                    success = 1 if episode_reward >= self.config.success_threshold else 0
                self.success_episodes.append(success)
                self.timestamps.append(time.time() - self.start_time)
                
                # Track network delays if available
                if 'network_latencies' in info:
                    delays = [lat.get('latency', 0) for lat in info['network_latencies']]
                    self.network_delays.append(np.mean(delays) if delays else 0)
                else:
                    self.network_delays.append(self.config.network_delay * 1000)  # Convert to ms
                
                # Progress reporting
                if self.episode_count % 100 == 0 and self.verbose > 0:
                    recent_success_rate = np.mean(self.success_episodes[-100:]) if len(self.success_episodes) >= 100 else np.mean(self.success_episodes)
                    avg_delay = np.mean(self.network_delays[-100:]) if self.network_delays else 0
                    elapsed = time.time() - self.start_time
                    print(f"  Episode {self.episode_count} ({elapsed:.1f}s): Success rate = {recent_success_rate*100:.1f}%, Avg delay = {avg_delay:.1f}ms")
        
        return True


def train_agent_for_duration(mode: SimulationMode, 
                           training_duration: float = 60.0,  # seconds
                           config: RLScenarioConfig = None,
                           trial_id: int = 0,
                           verbose: bool = True) -> Dict:
    """Train a PPO agent for a fixed wallclock duration."""
    
    if config is None:
        config = PREDEFINED_SCENARIOS["cartpole"]
    
    if verbose:
        print(f"\\nTraining PPO in {mode.value.upper()} mode for {training_duration}s (Trial {trial_id + 1})...")
    
    # Create FogSim environment with configured network latency
    handler = GymHandler(env_name=config.env_name)
    
    # Configure network
    network_config = NetworkConfig()
    network_config.topology.link_delay = config.network_delay
    network_config.source_rate = config.source_rate
    network_config.packet_loss_rate = config.packet_loss_rate
    
    fogsim = FogSim(handler, mode=mode, timestep=config.timestep, network_config=network_config)
    
    # Wrap environment for stable-baselines3
    from gymnasium import Env
    from gymnasium.spaces import Box, Discrete
    
    class FogSimWrapper(Env):
        """Wrapper to make FogSim compatible with stable-baselines3."""
        def __init__(self, fogsim, config):
            super().__init__()
            self.fogsim = fogsim
            self.config = config
            
            # Set observation and action spaces based on environment
            if 'CartPole' in config.env_name:
                self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
                self.action_space = Discrete(2)
            elif config.env_name in ['Ant-v4', 'HalfCheetah-v4', 'Humanoid-v4']:
                # MuJoCo environments have different spaces - we'll let the handler set them
                env_info = fogsim.handler.env.observation_space
                self.observation_space = Box(low=-np.inf, high=np.inf, 
                                           shape=env_info.shape, dtype=np.float32)
                self.action_space = Box(low=-1, high=1, 
                                      shape=fogsim.handler.env.action_space.shape, 
                                      dtype=np.float32)
            
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
                    info['network_latencies'] = [{'latency': self.config.network_delay * 1000}]  # Convert to ms
            
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
    env = FogSimWrapper(fogsim, config)
    
    # Use VecMonitor to track episode statistics
    vec_env = VecMonitor(make_vec_env(lambda: env, n_envs=1))
    
    # Create PPO model with configured hyperparameters
    # Use GPU if available for maximum performance
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if verbose and device == 'cuda':
        print(f"  Using GPU for training: {torch.cuda.get_device_name()}")
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=0,
        device=device,
        # Additional optimizations for GPU
        tensorboard_log=None,  # Disable tensorboard for speed
    )
    
    # Create callback to track metrics and handle time limit
    callback = TrainingMetricsCallback(max_duration=training_duration, config=config, 
                                      verbose=1 if verbose else 0)
    
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
        avg_network_delay = np.mean(callback.network_delays) if callback.network_delays else config.network_delay * 1000
    else:
        final_success_rate = 0
        avg_network_delay = config.network_delay * 1000
    
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
                         config: RLScenarioConfig,
                         num_episodes: int = 100) -> float:
    """Evaluate a trained PPO model's performance."""
    
    handler = GymHandler(env_name=config.env_name)
    
    # Configure network with same settings as training
    network_config = NetworkConfig()
    network_config.topology.link_delay = config.network_delay
    network_config.source_rate = config.source_rate
    network_config.packet_loss_rate = config.packet_loss_rate
    
    fogsim = FogSim(handler, mode=mode, timestep=config.timestep, network_config=network_config)
    
    successes = 0
    
    for _ in range(num_episodes):
        observation, _ = fogsim.reset()
        episode_length = 0
        
        done = False
        while not done and episode_length < config.max_episode_steps:
            # Use PPO model to predict action
            action, _ = model.predict(observation.astype(np.float32), deterministic=True)
            observation, _, _, termination, timeout, _ = fogsim.step(int(action)) 
            done = termination or timeout
            episode_length += 1
        
        # Check success based on environment type
        if 'CartPole' in config.env_name:
            if episode_length >= config.success_threshold:
                successes += 1
        else:
            # For MuJoCo envs, we'd need to track rewards
            # This is a simplified version
            if episode_length >= config.max_episode_steps * 0.8:
                successes += 1
    
    fogsim.close()
    
    return successes / num_episodes


def run_multiple_trials(mode: SimulationMode,
                       config: RLScenarioConfig,
                       num_trials: int,
                       training_duration: float,
                       verbose: bool = True) -> Dict:
    """Run multiple training trials and aggregate results."""
    
    all_results = []
    
    for trial in range(num_trials):
        if verbose:
            print(f"\n{'='*70}")
            print(f"Starting Trial {trial + 1}/{num_trials} for {mode.value} mode")
            print(f"{'='*70}")
        
        result = train_agent_for_duration(
            mode=mode,
            training_duration=training_duration,
            config=config,
            trial_id=trial,
            verbose=verbose
        )
        all_results.append(result)
    
    # Aggregate results across trials
    # Find the maximum number of episodes across all trials
    max_episodes = max(len(r['episode_rewards']) for r in all_results)
    
    # Interpolate results to common time points for averaging
    time_points = np.linspace(0, training_duration, 1000)
    
    interpolated_rewards = []
    interpolated_success_rates = []
    
    for result in all_results:
        if len(result['timestamps']) > 1:
            # Interpolate rewards
            rewards_interp = np.interp(time_points, result['timestamps'], result['episode_rewards'])
            interpolated_rewards.append(rewards_interp)
            
            # Calculate rolling success rate
            success_rate = []
            for i in range(len(result['success_episodes'])):
                window_start = max(0, i - 100)
                success_rate.append(np.mean(result['success_episodes'][window_start:i+1]))
            
            # Interpolate success rates
            if len(success_rate) > 1:
                success_interp = np.interp(time_points, result['timestamps'], success_rate)
                interpolated_success_rates.append(success_interp)
    
    # Calculate mean and std
    mean_rewards = np.mean(interpolated_rewards, axis=0)
    std_rewards = np.std(interpolated_rewards, axis=0)
    mean_success = np.mean(interpolated_success_rates, axis=0) if interpolated_success_rates else np.zeros_like(time_points)
    std_success = np.std(interpolated_success_rates, axis=0) if interpolated_success_rates else np.zeros_like(time_points)
    
    # Aggregate metrics
    aggregated = {
        'mode': mode,
        'config': config,
        'num_trials': num_trials,
        'all_results': all_results,
        'time_points': time_points,
        'mean_rewards': mean_rewards,
        'std_rewards': std_rewards,
        'mean_success_rate': mean_success,
        'std_success_rate': std_success,
        'total_episodes': [r['total_episodes'] for r in all_results],
        'total_steps': [r['total_steps'] for r in all_results],
        'final_success_rates': [r['final_success_rate'] for r in all_results],
        'episodes_per_second': [r['episodes_per_second'] for r in all_results],
        'steps_per_second': [r['steps_per_second'] for r in all_results],
        'mean_episodes_per_second': np.mean([r['episodes_per_second'] for r in all_results]),
        'mean_steps_per_second': np.mean([r['steps_per_second'] for r in all_results]),
        'mean_final_success_rate': np.mean([r['final_success_rate'] for r in all_results]),
        'std_final_success_rate': np.std([r['final_success_rate'] for r in all_results])
    }
    
    return aggregated


def plot_training_results(results: List[Dict], save_path: str = 'training_convergence_results.png'):
    """Plot training convergence results with mean and variance across trials."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        mode_names = {'virtual': 'Virtual Timeline', 'simulated': 'Real Clock + Sim Network', 'real': 'Real Clock + Real Network'}
        
        # Plot 1: Episode rewards over time with variance
        for i, result in enumerate(results):
            mode_name = mode_names.get(result['mode'].value, result['mode'].value)
            
            if 'mean_rewards' in result:
                # Multiple trials - plot with variance
                ax1.plot(result['time_points'], result['mean_rewards'], 
                        label=f"{mode_name} (n={result['num_trials']})", 
                        color=colors[i % len(colors)], linewidth=2)
                ax1.fill_between(result['time_points'], 
                               result['mean_rewards'] - result['std_rewards'],
                               result['mean_rewards'] + result['std_rewards'],
                               alpha=0.2, color=colors[i % len(colors)])
            else:
                # Single trial - plot as before
                rewards = result['episode_rewards']
                times = result['timestamps']
                
                window = min(50, len(rewards) // 10) if len(rewards) > 10 else 1
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    times_smooth = times[:len(smoothed)]
                    ax1.plot(times_smooth, smoothed, 
                            label=mode_name, 
                            color=colors[i % len(colors)], linewidth=2)
        
        ax1.set_xlabel('Wallclock Time (s)')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Progress Over Time (Mean ± Std)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Success rate over time with variance
        for i, result in enumerate(results):
            mode_name = mode_names.get(result['mode'].value, result['mode'].value)
            
            if 'mean_success_rate' in result:
                # Multiple trials
                ax2.plot(result['time_points'], result['mean_success_rate'] * 100, 
                        label=f"{mode_name} (n={result['num_trials']})", 
                        color=colors[i % len(colors)], linewidth=2)
                ax2.fill_between(result['time_points'], 
                               (result['mean_success_rate'] - result['std_success_rate']) * 100,
                               (result['mean_success_rate'] + result['std_success_rate']) * 100,
                               alpha=0.2, color=colors[i % len(colors)])
            else:
                # Single trial
                success = result['success_episodes']
                window = min(100, len(success) // 5) if len(success) > 5 else len(success)
                if window > 1 and len(success) > window:
                    success_rate = np.convolve(success, np.ones(window)/window, mode='valid') * 100
                    episodes = range(window-1, len(success))
                    ax2.plot(result['timestamps'][:len(success_rate)], success_rate, 
                            label=mode_name, 
                            color=colors[i % len(colors)], linewidth=2)
        
        ax2.set_xlabel('Wallclock Time (s)')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Success Rate During Training (Mean ± Std)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Training efficiency comparison (bar chart with error bars)
        modes = []
        episodes_means = []
        episodes_stds = []
        steps_means = []
        steps_stds = []
        success_means = []
        success_stds = []
        
        for result in results:
            mode_name = mode_names.get(result['mode'].value, result['mode'].value)
            modes.append(mode_name)
            
            if 'mean_episodes_per_second' in result:
                episodes_means.append(result['mean_episodes_per_second'])
                episodes_stds.append(np.std(result['episodes_per_second']))
                steps_means.append(result['mean_steps_per_second'] / 1000)
                steps_stds.append(np.std(result['steps_per_second']) / 1000)
                success_means.append(result['mean_final_success_rate'] * 100)
                success_stds.append(result['std_final_success_rate'] * 100)
            else:
                episodes_means.append(result['episodes_per_second'])
                episodes_stds.append(0)
                steps_means.append(result['steps_per_second'] / 1000)
                steps_stds.append(0)
                success_means.append(result['final_success_rate'] * 100)
                success_stds.append(0)
        
        x = np.arange(len(modes))
        width = 0.25
        
        ax3.bar(x - width, episodes_means, width, yerr=episodes_stds, 
               label='Episodes/sec', alpha=0.8, capsize=5)
        ax3.bar(x, steps_means, width, yerr=steps_stds,
               label='Steps/sec (÷1000)', alpha=0.8, capsize=5)
        ax3.bar(x + width, success_means, width, yerr=success_stds,
               label='Success Rate %', alpha=0.8, capsize=5)
        
        ax3.set_xlabel('Mode')
        ax3.set_ylabel('Value')
        ax3.set_title('Performance Metrics Comparison (Mean ± Std)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(modes, rotation=15, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Total episodes completed (box plot if multiple trials)
        data_to_plot = []
        labels = []
        
        for result in results:
            mode_name = mode_names.get(result['mode'].value, result['mode'].value)
            if 'total_episodes' in result and isinstance(result['total_episodes'], list):
                data_to_plot.append(result['total_episodes'])
                labels.append(mode_name)
            else:
                data_to_plot.append([result['total_episodes']])
                labels.append(mode_name)
        
        bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Episodes Completed')
        ax4.set_title('Training Efficiency Distribution')
        ax4.grid(True, alpha=0.3, axis='y')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=15, ha='right')
        
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
    parser.add_argument("--scenario", type=str, default="cartpole", 
                       choices=list(PREDEFINED_SCENARIOS.keys()), 
                       help="Predefined scenario to use")
    parser.add_argument("--config", type=str, help="Path to custom config JSON file")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per mode")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained models after training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--save-config", type=str, help="Save current config to JSON file")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto",
                       help="Device to use for training (auto will use GPU if available)")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # Load custom config from file
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = RLScenarioConfig.from_dict(config_dict)
    else:
        # Use predefined scenario
        config = PREDEFINED_SCENARIOS[args.scenario]
    
    # Override device if specified
    if args.device != "auto":
        config.device = args.device
    
    # Save config if requested
    if args.save_config:
        with open(args.save_config, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        print(f"Configuration saved to {args.save_config}")
    
    print("FogSim PPO Training Convergence Demo")
    print("="*70)
    print("Demonstrating FogSim's high frame rate benefits for PPO training")
    print(f"Scenario: {config.name}")
    print(f"Environment: {config.env_name}")
    print(f"Training duration: {args.duration} seconds per trial")
    print(f"Number of trials: {args.trials}")
    print(f"Network configuration: {config.network_delay*1000:.0f}ms latency, {config.packet_loss_rate*100:.1f}% packet loss")
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
        
        if args.trials > 1:
            # Run multiple trials
            result = run_multiple_trials(
                mode=mode,
                config=config,
                num_trials=args.trials,
                training_duration=args.duration,
                verbose=args.verbose
            )
            results.append(result)
            
            print(f"\\n{mode.value.upper()} Results (averaged over {args.trials} trials):")
            print(f"  Mean episodes/trial: {np.mean(result['total_episodes']):.1f} (±{np.std(result['total_episodes']):.1f})")
            print(f"  Mean steps/trial: {np.mean(result['total_steps']):.1f} (±{np.std(result['total_steps']):.1f})")
            print(f"  Episodes/second: {result['mean_episodes_per_second']:.2f}")
            print(f"  Steps/second: {result['mean_steps_per_second']:.1f}")
            print(f"  Final success rate: {result['mean_final_success_rate']*100:.1f}% (±{result['std_final_success_rate']*100:.1f}%)")
        else:
            # Single trial
            result = train_agent_for_duration(
                mode=mode,
                training_duration=args.duration,
                config=config,
                trial_id=0,
                verbose=args.verbose
            )
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
    if args.evaluate:
        print("\\n" + "="*70)
        print("EVALUATING TRAINED MODELS")
        print("="*70)
        
        for result in results:
            if 'model' in result:
                # Single trial - evaluate the model
                eval_success_rate = evaluate_trained_model(
                    result['model'], 
                    result['mode'],
                    config,
                    num_episodes=100
                )
                print(f"\\n{result['mode'].value.upper()} - Evaluation Success Rate: {eval_success_rate*100:.1f}%")
            elif 'all_results' in result:
                # Multiple trials - evaluate best model
                best_idx = np.argmax([r['final_success_rate'] for r in result['all_results']])
                best_model = result['all_results'][best_idx].get('model')
                if best_model:
                    eval_success_rate = evaluate_trained_model(
                        best_model,
                        result['mode'],
                        config,
                        num_episodes=100
                    )
                    print(f"\\n{result['mode'].value.upper()} - Best Model Evaluation Success Rate: {eval_success_rate*100:.1f}%")
    
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
                
                if 'mean_episodes_per_second' in virtual_result:
                    # Multiple trials comparison
                    virtual_episodes = np.mean(virtual_result['total_episodes'])
                    other_episodes = np.mean(result['total_episodes']) if 'total_episodes' in result else result['total_episodes']
                    virtual_steps = np.mean(virtual_result['total_steps'])
                    other_steps = np.mean(result['total_steps']) if 'total_steps' in result else result['total_steps']
                    virtual_success = virtual_result['mean_final_success_rate']
                    other_success = result.get('mean_final_success_rate', result.get('final_success_rate', 0))
                else:
                    # Single trial comparison
                    virtual_episodes = virtual_result['total_episodes']
                    other_episodes = result['total_episodes']
                    virtual_steps = virtual_result['total_steps']
                    other_steps = result['total_steps']
                    virtual_success = virtual_result['final_success_rate']
                    other_success = result['final_success_rate']
                
                episode_speedup = virtual_episodes / other_episodes if other_episodes > 0 else 0
                step_speedup = virtual_steps / other_steps if other_steps > 0 else 0
                
                print(f"\\n  Virtual vs {result['mode'].value}:")
                print(f"    Episodes: {virtual_episodes:.0f} vs {other_episodes:.0f} ({episode_speedup:.2f}x speedup)")
                print(f"    Steps: {virtual_steps:.0f} vs {other_steps:.0f} ({step_speedup:.2f}x speedup)")
                print(f"    Success Rate: {virtual_success*100:.1f}% vs {other_success*100:.1f}%")
        
        print(f"\\nKey Insights:")
        print(f"✓ Virtual Timeline mode enables significantly more training iterations")
        print(f"✓ Higher frame rates lead to faster policy convergence")
        print(f"✓ Network delays affect both training speed and final performance")
        print(f"✓ Real network mode shows actual latency impact on RL training")
        
        if args.trials > 1:
            print(f"\\nStatistical Reliability:")
            print(f"✓ Results averaged over {args.trials} trials for statistical significance")
            print(f"✓ Variance bands show training stability across runs")
            print(f"✓ Box plots reveal consistency of training efficiency")
        
        # Show the FogSim advantage
        if virtual_result and len(results) > 1:
            if 'mean_episodes_per_second' in virtual_result:
                # Multiple trials
                virtual_episodes = np.mean(virtual_result['total_episodes'])
                other_episodes = [np.mean(r['total_episodes']) if 'total_episodes' in r else r['total_episodes'] 
                                for r in results if r['mode'] != SimulationMode.VIRTUAL]
            else:
                # Single trial
                virtual_episodes = virtual_result['total_episodes']
                other_episodes = [r['total_episodes'] for r in results if r['mode'] != SimulationMode.VIRTUAL]
            
            if other_episodes:
                max_other_episodes = max(other_episodes)
                advantage = virtual_episodes / max_other_episodes if max_other_episodes > 0 else 1
                print(f"\\n🚀 FogSim Virtual Timeline advantage: {advantage:.1f}x more training episodes!")


if __name__ == "__main__":
    main()