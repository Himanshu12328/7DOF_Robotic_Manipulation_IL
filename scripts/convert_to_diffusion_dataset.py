import h5py
import os
import numpy as np
from datasets import Dataset, DatasetDict

# Path to your existing HDF5 dataset
hdf5_path = 'data/ik_demos.hdf5'

# Initialize lists to store the data
observations = []
actions = []

# Open the HDF5 file and extract data
with h5py.File(hdf5_path, 'r') as f:
    for key in f.keys():
        if 'states' in key:
            idx = key.split('_')[-1]
            obs = f[f'states_{idx}'][:]
            act = f[f'actions_{idx}'][:]
            for o, a in zip(obs, act):
                observations.append({'state': o.tolist()})
                actions.append({'action': a.tolist()})

# Combine observations and actions
data = []
for obs, act in zip(observations, actions):
    data.append({**obs, **act})

# Create a Hugging Face Dataset
dataset = Dataset.from_list(data)

# Optionally, split into train and test sets
split_dataset = dataset.train_test_split(test_size=0.1)

# Save the dataset locally
split_dataset.save_to_disk('lerobot_dataset')

print("Dataset conversion complete. Saved to 'lerobot_dataset'.")