"""
Evaluation Framework for FogSim

This module provides tools for evaluating FogSim's performance,
reproducibility, and sim-to-real gap across different modes.
"""

import time
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict
import json
import hashlib

from .time_backend import SimulationMode
from .env import Env


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a simulation run."""
    mode: SimulationMode
    episode_id: int
    
    # Timing metrics
    wallclock_time: float = 0.0
    simulation_time: float = 0.0
    frame_rate: float = 0.0
    frame_times: List[float] = field(default_factory=list)
    
    # Network metrics
    network_delays: List[float] = field(default_factory=list)
    packet_losses: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    
    # Task metrics
    total_reward: float = 0.0
    success: bool = False
    steps: int = 0
    
    # Reproducibility
    action_sequence: List[Any] = field(default_factory=list)
    observation_sequence: List[Any] = field(default_factory=list)
    state_hash: Optional[str] = None


@dataclass
class EvaluationResults:
    """Aggregated evaluation results."""
    mode: SimulationMode
    num_episodes: int
    
    # Performance
    avg_frame_rate: float = 0.0
    std_frame_rate: float = 0.0
    avg_wallclock_time: float = 0.0
    
    # Network
    avg_network_delay: float = 0.0
    std_network_delay: float = 0.0
    packet_loss_rate: float = 0.0
    
    # Task performance
    avg_reward: float = 0.0
    std_reward: float = 0.0
    success_rate: float = 0.0
    
    # Reproducibility
    deterministic: bool = False
    state_variance: float = 0.0
    
    # Raw data
    episodes: List[PerformanceMetrics] = field(default_factory=list)


class PerformanceTracker:
    """Tracks performance metrics during simulation."""
    
    def __init__(self, mode: SimulationMode, episode_id: int):
        self.metrics = PerformanceMetrics(mode=mode, episode_id=episode_id)
        self.start_wallclock = time.time()
        self.last_frame_time = self.start_wallclock
        
    def record_step(self, observation: np.ndarray, action: Any, 
                   reward: float, info: Dict[str, Any]) -> None:
        """Record metrics for a single step."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.metrics.frame_times.append(frame_time)
        self.metrics.total_reward += reward
        self.metrics.steps += 1
        
        # Record action and observation for reproducibility
        self.metrics.action_sequence.append(action)
        self.metrics.observation_sequence.append(observation.tolist() if isinstance(observation, np.ndarray) else observation)
        
        # Extract network metrics from info
        if 'network_latencies' in info:
            for latency_info in info['network_latencies']:
                self.metrics.network_delays.append(latency_info['latency'])
        
        if 'num_messages_received' in info:
            self.metrics.messages_received += info['num_messages_received']
    
    def finalize(self, success: bool, final_time: float) -> PerformanceMetrics:
        """Finalize metrics after episode completion."""
        self.metrics.wallclock_time = time.time() - self.start_wallclock
        self.metrics.simulation_time = final_time
        self.metrics.success = success
        
        if self.metrics.frame_times:
            self.metrics.frame_rate = 1.0 / np.mean(self.metrics.frame_times)
        
        # Generate state hash for reproducibility checking
        state_data = {
            'actions': self.metrics.action_sequence,
            'observations': self.metrics.observation_sequence,
            'reward': self.metrics.total_reward,
            'steps': self.metrics.steps
        }
        state_str = json.dumps(state_data, sort_keys=True)
        self.metrics.state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        
        return self.metrics


class FogSimEvaluator:
    """Main evaluation class for FogSim experiments."""
    
    def __init__(self):
        self.results: Dict[SimulationMode, EvaluationResults] = {}
    
    def evaluate_mode(self, env_factory: Callable[[], Env], 
                     mode: SimulationMode,
                     num_episodes: int = 10,
                     episode_length: int = 200,
                     policy: Optional[Callable] = None) -> EvaluationResults:
        """
        Evaluate FogSim in a specific mode.
        
        Args:
            env_factory: Factory function to create environment
            mode: Simulation mode to evaluate
            num_episodes: Number of episodes to run
            episode_length: Maximum steps per episode
            policy: Optional policy function (defaults to random)
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating {mode.value} mode with {num_episodes} episodes")
        
        results = EvaluationResults(mode=mode, num_episodes=num_episodes)
        episode_metrics = []
        
        for episode in range(num_episodes):
            # Create environment
            env = env_factory()
            tracker = PerformanceTracker(mode, episode)
            
            # Run episode
            obs, info = env.reset()
            
            for step in range(episode_length):
                # Get action from policy
                if policy:
                    action = policy(obs)
                else:
                    # Random policy
                    action = env.handler.action_space.sample()
                
                # Step environment
                obs, reward, success, termination, timeout, info = env.step(action)
                
                # Track metrics
                tracker.record_step(obs, action, reward, info)
                
                if termination or timeout:
                    break
            
            # Finalize metrics
            metrics = tracker.finalize(success, env.time_manager.now())
            episode_metrics.append(metrics)
            
            logger.info(f"Episode {episode + 1}: Reward={metrics.total_reward:.2f}, "
                       f"FPS={metrics.frame_rate:.1f}, Success={metrics.success}")
            
            env.close()
        
        # Aggregate results
        results.episodes = episode_metrics
        self._aggregate_results(results)
        
        # Check reproducibility
        self._check_reproducibility(results)
        
        self.results[mode] = results
        return results
    
    def _aggregate_results(self, results: EvaluationResults) -> None:
        """Aggregate episode metrics into summary statistics."""
        if not results.episodes:
            return
        
        # Performance metrics
        frame_rates = [ep.frame_rate for ep in results.episodes]
        results.avg_frame_rate = np.mean(frame_rates)
        results.std_frame_rate = np.std(frame_rates)
        
        wallclock_times = [ep.wallclock_time for ep in results.episodes]
        results.avg_wallclock_time = np.mean(wallclock_times)
        
        # Network metrics
        all_delays = []
        for ep in results.episodes:
            all_delays.extend(ep.network_delays)
        
        if all_delays:
            results.avg_network_delay = np.mean(all_delays) * 1000  # Convert to ms
            results.std_network_delay = np.std(all_delays) * 1000
        
        # Task performance
        rewards = [ep.total_reward for ep in results.episodes]
        results.avg_reward = np.mean(rewards)
        results.std_reward = np.std(rewards)
        
        successes = [ep.success for ep in results.episodes]
        results.success_rate = np.mean(successes)
    
    def _check_reproducibility(self, results: EvaluationResults) -> None:
        """Check if results are reproducible."""
        if results.mode != SimulationMode.VIRTUAL:
            # Only check reproducibility for virtual mode
            return
        
        # Check if all episodes have the same state hash
        state_hashes = [ep.state_hash for ep in results.episodes]
        unique_hashes = set(state_hashes)
        
        results.deterministic = len(unique_hashes) == 1
        
        # Calculate variance in rewards as a measure of reproducibility
        rewards = [ep.total_reward for ep in results.episodes]
        results.state_variance = np.var(rewards)
        
        logger.info(f"Reproducibility check: Deterministic={results.deterministic}, "
                   f"Unique states={len(unique_hashes)}, Variance={results.state_variance:.6f}")
    
    def compare_modes(self, env_factory: Callable[[], Env],
                     num_episodes: int = 10,
                     episode_length: int = 200) -> Dict[SimulationMode, EvaluationResults]:
        """
        Compare all three simulation modes.
        
        Args:
            env_factory: Factory function to create environment
            num_episodes: Number of episodes per mode
            episode_length: Maximum steps per episode
            
        Returns:
            Dictionary of results for each mode
        """
        for mode in SimulationMode:
            # Skip real network mode if not running as root
            if mode == SimulationMode.REAL_NET:
                try:
                    import os
                    if hasattr(os, 'geteuid') and os.geteuid() != 0:
                        logger.warning("Skipping REAL_NET mode - requires root privileges")
                        continue
                except:
                    logger.warning("Skipping REAL_NET mode - not supported on this platform")
                    continue
            
            # Create mode-specific environment factory
            def mode_env_factory():
                env = env_factory()
                env.mode = mode
                env.time_manager.mode = mode
                return env
            
            self.evaluate_mode(mode_env_factory, mode, num_episodes, episode_length)
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a comparison report of all evaluated modes."""
        report = []
        report.append("="*70)
        report.append("FogSim Evaluation Report")
        report.append("="*70)
        
        for mode, results in self.results.items():
            report.append(f"\n{mode.value.upper()} MODE")
            report.append("-"*40)
            
            # Performance
            report.append(f"Performance:")
            report.append(f"  Average FPS: {results.avg_frame_rate:.1f} ± {results.std_frame_rate:.1f}")
            report.append(f"  Average Episode Time: {results.avg_wallclock_time:.2f}s")
            
            # Network
            if results.avg_network_delay > 0:
                report.append(f"\nNetwork:")
                report.append(f"  Average Delay: {results.avg_network_delay:.1f} ± {results.std_network_delay:.1f}ms")
            
            # Task Performance
            report.append(f"\nTask Performance:")
            report.append(f"  Average Reward: {results.avg_reward:.2f} ± {results.std_reward:.2f}")
            report.append(f"  Success Rate: {results.success_rate*100:.1f}%")
            
            # Reproducibility
            if mode == SimulationMode.VIRTUAL:
                report.append(f"\nReproducibility:")
                report.append(f"  Deterministic: {results.deterministic}")
                report.append(f"  State Variance: {results.state_variance:.6f}")
        
        # Comparison
        if len(self.results) > 1:
            report.append(f"\n{'='*70}")
            report.append("COMPARISON")
            report.append("="*70)
            
            # Frame rate comparison
            report.append("\nFrame Rate (FPS):")
            for mode, results in self.results.items():
                speedup = results.avg_frame_rate / list(self.results.values())[0].avg_frame_rate
                report.append(f"  {mode.value}: {results.avg_frame_rate:.1f} ({speedup:.2f}x)")
        
        return "\n".join(report)
    
    def save_results(self, filename: str) -> None:
        """Save evaluation results to JSON file."""
        data = {}
        for mode, results in self.results.items():
            data[mode.value] = {
                'num_episodes': results.num_episodes,
                'avg_frame_rate': results.avg_frame_rate,
                'avg_reward': results.avg_reward,
                'success_rate': results.success_rate,
                'deterministic': results.deterministic,
                'episodes': [
                    {
                        'episode_id': ep.episode_id,
                        'wallclock_time': ep.wallclock_time,
                        'total_reward': ep.total_reward,
                        'success': ep.success,
                        'steps': ep.steps
                    }
                    for ep in results.episodes
                ]
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filename}")