import pandas as pd

# Read the CSV data
df = pd.read_csv('monte_carlo_results/statistics.csv')

# The column name has a space, so we need to use ' delta_k' instead of 'delta_k'
collision_rates = df.groupby('delta_k')['collision'].agg(['count', 'sum'])
collision_rates['rate'] = (collision_rates['sum'] / collision_rates['count']) * 100

print("\nCollision Rates by delta_k:")
print("------------------------")
for dk, row in collision_rates.iterrows():
    print(f"delta_k = {dk:2d}: {row['rate']:5.1f}% ({int(row['sum'])}/{int(row['count'])} collisions)")