import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# Set seaborn style
sns.set_theme(style="whitegrid")
# sns.set_palette("")
sns.set_context("poster")

# Read from different results_* directories and create separate plots
results_dirs = glob.glob('results_*')
for results_dir in results_dirs:
    # Create a new figure for each directory
    plt.figure(figsize=(10, 6))
    
    # Read all CSV files in the current directory
    files = glob.glob(f'{results_dir}/collision_probabilities_*.csv')
    print(f"Processing files in {results_dir}:", files)
    
    # Create ground truth data (delta_k = 1)
    first_file = pd.read_csv(f"{results_dir}/collision_probabilities_1.csv")
    ground_truth = pd.DataFrame({
        'timestamp': first_file['timestamp'],
        'collision_probability': first_file['collision_probability']
    })
    
    # Find the first timestamp where collision probability is 0
    start_time = ground_truth[ground_truth['collision_probability'] > 0].iloc[0]['timestamp']
    
    # Filter ground truth data to start from this timestamp
    ground_truth = ground_truth[ground_truth['timestamp'] >= start_time]
    
    # Plot ground truth
    plt.plot(ground_truth['timestamp'], ground_truth['collision_probability'], 
             'k--', label='Ground Truth', linewidth=2)
    
    files.remove(f"{results_dir}/collision_probabilities_1.csv")
    
    # Sort files by delta_k value
    file_dk_pairs = []
    for file in files:
        df = pd.read_csv(file)
        df = df[df['timestamp'] >= ground_truth['timestamp'].iloc[0]]
        delta_k = df['delta_k'].iloc[0]
        latency = delta_k * 10
        file_dk_pairs.append((file, latency))
    
    file_dk_pairs.sort(key=lambda x: x[1])
    
    # Create a color map for consistent colors across different plots
    unique_latencies = sorted(list(set(latency for _, latency in file_dk_pairs)))
    color_map = dict(zip(unique_latencies, sns.color_palette("husl", len(unique_latencies))))
    
    # Plot collision probabilities
    for file, latency in file_dk_pairs:
        df = pd.read_csv(file)
        # Filter data to start from the same timestamp as ground truth
        df = df[df['timestamp'] >= start_time]
        print(df.head())
        sns.lineplot(data=df, x='timestamp', y='collision_probability', 
                    label=f'{latency}ms', color=color_map[latency], legend=False)
    
    plt.xlabel('Time (s)')
    plt.ylabel('Perceived \n Collision Risk')
    
    # Move y-axis to the right
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    
    # Move legend outside the plot
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
    #           frameon=True, fancybox=True, shadow=True)
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    # plt.subplots_adjust(right=0.85)  # Make room for the legend
    
    # Save figure with directory name in the filename
    dir_name = results_dir.split('/')[-1]
    plt.savefig(f'{results_dir}/collision_probabilities_{dir_name}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

