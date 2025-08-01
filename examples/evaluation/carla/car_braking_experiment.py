"""
Car Braking Experiment - Demonstrating Reproducibility Issues

This experiment implements the car braking scenario from CLAUDE.md:
- A car brakes at X meters away from an obstacle
- Shows how relying on external time leads to reproducibility issues
- Compares variance across different modes
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import Dict, List
from dataclasses import dataclass

from fogsim import SimulationMode


@dataclass
class BrakingScenario:
    """Configuration for a braking scenario."""
    initial_velocity: float = 20.0  # m/s
    braking_distance: float = 50.0  # meters from obstacle
    obstacle_position: float = 100.0  # meters
    max_deceleration: float = 8.0  # m/s^2
    timestep: float = 0.01  # seconds


class SimpleCar:
    """Simple 1D car dynamics model."""
    
    def __init__(self, scenario: BrakingScenario):
        self.scenario = scenario
        self.position = 0.0
        self.velocity = scenario.initial_velocity
        self.is_braking = False
        self.collision = False
        self.stopped = False
        
    def update(self, action: float, dt: float):
        """Update car state. Action: 0=coast, 1=brake"""
        if self.stopped or self.collision:
            return
            
        # Apply braking if commanded
        if action > 0.5:
            acceleration = -self.scenario.max_deceleration
        else:
            acceleration = 0.0
            
        # Update velocity and position
        self.velocity = max(0.0, self.velocity + acceleration * dt)
        self.position += self.velocity * dt
        
        # Check for collision
        if self.position >= self.scenario.obstacle_position:
            self.collision = True
            
        # Check if stopped
        if self.velocity == 0.0:
            self.stopped = True
            
    def should_brake(self) -> bool:
        """Determine if car should brake based on distance to obstacle."""
        distance_to_obstacle = self.scenario.obstacle_position - self.position
        return distance_to_obstacle <= self.scenario.braking_distance
        
    def get_state(self) -> Dict:
        """Get current car state."""
        return {
            'position': self.position,
            'velocity': self.velocity,
            'distance_to_obstacle': self.scenario.obstacle_position - self.position,
            'is_braking': self.is_braking,
            'collision': self.collision,
            'stopped': self.stopped
        }


def run_braking_scenario(mode: SimulationMode, scenario: BrakingScenario, 
                        num_runs: int = 10) -> Dict[str, List[float]]:
    """Run the braking scenario multiple times in a given mode."""
    
    results = {
        'final_positions': [],
        'stopping_distances': [],
        'collisions': [],
        'frame_times': []
    }
    
    for run in range(num_runs):
        # Create a simple car simulation
        car = SimpleCar(scenario)
        
        # Track timing for each step
        step_times = []
        positions_over_time = []
        
        # Run simulation
        start_time = time.time()
        last_step_time = start_time
        
        while not (car.stopped or car.collision) and car.position < scenario.obstacle_position + 10:
            # Decide action based on distance
            action = 1.0 if car.should_brake() else 0.0
            
            # Simulate network delay in non-virtual modes
            if mode != SimulationMode.VIRTUAL:
                # Add small random delay to simulate timing variations
                delay = np.random.normal(0.001, 0.0002)  # 1ms ± 0.2ms
                time.sleep(max(0, delay))
            
            # Update car with actual elapsed time (for non-virtual modes)
            current_time = time.time()
            if mode == SimulationMode.VIRTUAL:
                dt = scenario.timestep  # Fixed timestep
            else:
                dt = current_time - last_step_time  # Actual elapsed time
                
            car.update(action, dt)
            
            # Record timing
            step_times.append(current_time - last_step_time)
            positions_over_time.append(car.position)
            last_step_time = current_time
        
        # Record results
        final_state = car.get_state()
        results['final_positions'].append(final_state['position'])
        results['stopping_distances'].append(
            scenario.obstacle_position - final_state['position']
        )
        results['collisions'].append(1.0 if final_state['collision'] else 0.0)
        results['frame_times'].extend(step_times)
        
        print(f"  Run {run + 1}: Final position = {final_state['position']:.3f}m, "
              f"Collision = {final_state['collision']}")
    
    return results


def analyze_variance(results: Dict[str, List[float]]) -> Dict[str, float]:
    """Analyze variance in results."""
    return {
        'position_variance': np.var(results['final_positions']),
        'position_std': np.std(results['final_positions']),
        'stopping_distance_variance': np.var(results['stopping_distances']),
        'stopping_distance_std': np.std(results['stopping_distances']),
        'collision_rate': np.mean(results['collisions']),
        'timing_variance': np.var(results['frame_times'])
    }


def plot_braking_results(all_results: Dict[SimulationMode, Dict]):
    """Plot the braking experiment results."""
    try:
        _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        modes = list(all_results.keys())
        mode_names = [m.value for m in modes]
        
        # Plot 1: Final position variance
        variances = [all_results[m]['analysis']['position_variance'] for m in modes]
        ax1.bar(mode_names, variances)
        ax1.set_ylabel('Position Variance (m²)')
        ax1.set_title('Final Position Variance by Mode')
        ax1.set_yscale('log')
        
        # Plot 2: Box plot of final positions
        positions_data = [all_results[m]['results']['final_positions'] for m in modes]
        ax2.boxplot(positions_data, labels=mode_names)
        ax2.set_ylabel('Final Position (m)')
        ax2.set_title('Distribution of Final Positions')
        ax2.axhline(y=100, color='r', linestyle='--', label='Obstacle')
        ax2.legend()
        
        # Plot 3: Collision rates
        collision_rates = [all_results[m]['analysis']['collision_rate'] * 100 for m in modes]
        ax3.bar(mode_names, collision_rates)
        ax3.set_ylabel('Collision Rate (%)')
        ax3.set_title('Collision Rate by Mode')
        
        # Plot 4: Timing variance
        timing_vars = [all_results[m]['analysis']['timing_variance'] * 1e6 for m in modes]
        ax4.bar(mode_names, timing_vars)
        ax4.set_ylabel('Timing Variance (μs²)')
        ax4.set_title('Frame Time Variance')
        ax4.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig('car_braking_results.png')
        print("\nResults saved to car_braking_results.png")
        
    except Exception as e:
        print(f"\nCould not generate plots: {e}")


def main():
    """Run the car braking experiment."""
    print("Car Braking Experiment")
    print("="*70)
    print("Demonstrating how wallclock dependency affects reproducibility")
    print("Scenario: Car traveling at 20 m/s must brake 50m before obstacle at 100m")
    
    # Define scenario
    scenario = BrakingScenario()
    
    # Test different modes
    all_results = {}
    
    for mode in [SimulationMode.VIRTUAL, SimulationMode.SIMULATED_NET]:
        print(f"\n{mode.value.upper()} Mode:")
        print("-"*40)
        
        results = run_braking_scenario(mode, scenario, num_runs=20)
        analysis = analyze_variance(results)
        
        all_results[mode] = {
            'results': results,
            'analysis': analysis
        }
        
        print(f"\nAnalysis:")
        print(f"  Position variance: {analysis['position_variance']:.6f} m²")
        print(f"  Position std dev: {analysis['position_std']:.3f} m")
        print(f"  Collision rate: {analysis['collision_rate']*100:.1f}%")
        print(f"  Timing variance: {analysis['timing_variance']*1e6:.3f} μs²")
        
        if mode == SimulationMode.VIRTUAL and analysis['position_variance'] < 1e-10:
            print("  ✓ PERFECT REPRODUCIBILITY - All runs identical!")
    
    # Generate plots
    plot_braking_results(all_results)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    virtual_var = all_results[SimulationMode.VIRTUAL]['analysis']['position_variance']
    simulated_var = all_results[SimulationMode.SIMULATED_NET]['analysis']['position_variance']
    
    print(f"\nPosition Variance:")
    print(f"  Virtual Mode: {virtual_var:.10f} m²")
    print(f"  Simulated Mode: {simulated_var:.6f} m²")
    print(f"  Variance Ratio: {simulated_var/max(virtual_var, 1e-10):.1f}x")
    
    print(f"\nCollision Rates:")
    for mode in all_results:
        rate = all_results[mode]['analysis']['collision_rate']
        print(f"  {mode.value}: {rate*100:.1f}%")
    
    print("\n✓ Experiment demonstrates that virtual timeline provides")
    print("  perfect reproducibility while wallclock-based modes show variance!")


if __name__ == "__main__":
    main()