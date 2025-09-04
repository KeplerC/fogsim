import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_trajectory_data(filename):
    """Load trajectory data from CSV file"""
    data = pd.read_csv(filename, header=None, names=['x', 'y', 'z'])
    return data

def plot_trajectories():
    """Plot both obstacle and ego vehicle trajectories"""
    # Load trajectory data
    obstacle_data = load_trajectory_data('examples/evaluation/collision_probability/obstacle_trajectory_sample_0.csv')
    ego_data = load_trajectory_data('examples/evaluation/collision_probability/ego_trajectory_sample_0.csv')
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot trajectory lines
    ax.plot(obstacle_data['x'], obstacle_data['y'], 'r-', linewidth=2, alpha=0.7, label='Obstacle Path')
    ax.plot(ego_data['x'], ego_data['y'], 'b-', linewidth=2, alpha=0.7, label='Ego Vehicle Path')
    
    # Plot center points (dots) - sample every 15th point for visibility
    dot_interval = 5
    obstacle_sample = obstacle_data.iloc[::dot_interval]
    ego_sample = ego_data.iloc[::dot_interval]
    
    ax.scatter(obstacle_sample['x'], obstacle_sample['y'], c='red', s=40, alpha=0.8, zorder=5, marker='o', edgecolors='darkred')
    ax.scatter(ego_sample['x'], ego_sample['y'], c='blue', s=40, alpha=0.8, zorder=5, marker='o', edgecolors='darkblue')
    
    # Set equal aspect ratio and adjust limits
    ax.set_aspect('equal')
    
    # Calculate bounds for both trajectories
    all_x = np.concatenate([obstacle_data['x'], ego_data['x']])
    all_y = np.concatenate([obstacle_data['y'], ego_data['y']])
    
    margin = 10  # Add some margin around the trajectories
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Add labels and title
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Vehicle Trajectories\nRed: Obstacle, Blue: Ego Vehicle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add text annotation
    ax.text(0.02, 0.98, 'Dots represent vehicle center points along the trajectory', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Print some statistics
    print(f"Obstacle trajectory: {len(obstacle_data)} points")
    print(f"Ego vehicle trajectory: {len(ego_data)} points")
    
    plt.tight_layout()
    plt.show()
    
    # Save the plot
    plt.savefig('trajectory_visualization.png', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'trajectory_visualization.png'")

if __name__ == "__main__":
    plot_trajectories() 