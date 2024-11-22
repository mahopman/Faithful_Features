import pandas as pd
import os
import pickle

# Create output directory if it doesn't exist
output_dir = "results/contrast_experiments"

# Check if progress files exist
experiments_file = f"{output_dir}/experiments_results.csv"
features_file = f"{output_dir}/features_results.csv"
neighbors_file = f"{output_dir}/feature_nearest_neighbors.pkl"

experiments_df = pd.read_csv(experiments_file)

features_df = pd.read_csv(features_file)
with open(neighbors_file, "rb") as f:
    feature_nearest_neighbors = pickle.load(f)

for key, value in feature_nearest_neighbors.items():
    if 'inappropriate' in str(key):
        print(key)
        print(value)
