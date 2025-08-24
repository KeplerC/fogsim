import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import numpy as np

# Set seaborn style
sns.set_theme(style="whitegrid")
# sns.set_palette("")
sns.set_context("poster")

# Read all CSV files in the folder
files = glob.glob('collision_probabilities_*.csv')
plt.figure(figsize=(10, 6))

# Create ground truth data (delta_k = 1)
print(files)
first_file = pd.read_csv("collision_probabilities_1.csv")  # Get timestamp from any file
ground_truth = pd.DataFrame({
    'timestamp': first_file['timestamp'],
    'collision_probability': first_file['collision_probability']
})

# Plot all lines including ground truth
plt.plot(ground_truth['timestamp'], ground_truth['collision_probability'], 
         'k--', label='Ground Truth', linewidth=2)

files.remove("collision_probabilities_1.csv")

# Sort files by delta_k value
file_dk_pairs = []
for file in files:
    df = pd.read_csv(file)
    # Trim other dataframes to start from the same timestamp as ground truth
    df = df[df['timestamp'] >= ground_truth['timestamp'].iloc[0]]
    delta_k = df['delta_k'].iloc[0]
    # Convert delta_k to latency (multiply by 10 since delta_k=1 corresponds to 10ms)
    latency = delta_k * 10
    file_dk_pairs.append((file, latency))

file_dk_pairs.sort(key=lambda x: x[1])  # Sort by latency

# Plot collision probabilities
for file, latency in file_dk_pairs:
    df = pd.read_csv(file)
    sns.lineplot(data=df, x='timestamp', y='collision_probability', label=f'{latency}ms')

plt.xlabel('Time (s)')
plt.ylabel('Perceived \n Collision Probability')
plt.legend(frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('collision_probabilities.pdf', dpi=300, bbox_inches='tight')

