import h5py
import numpy as np
import os

# Dynamically get the base directory relative to this script
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def collect_demo_data(num_demos=5, demo_length=100):
    demonstrations = []
    for i in range(num_demos):
        states = np.random.rand(demo_length, 7)   # Simulated joint states
        actions = np.random.rand(demo_length, 7)  # Simulated actions
        demonstrations.append((states, actions))

    file_path = os.path.join(DATA_DIR, 'demos.hdf5')
    with h5py.File(file_path, 'w') as f:
        for idx, (s, a) in enumerate(demonstrations):
            f.create_dataset(f'states_{idx}', data=s)
            f.create_dataset(f'actions_{idx}', data=a)

    print(f"Demonstration data saved at: {file_path}")

if __name__ == "__main__":
    try:
        collect_demo_data()
    except Exception as e:
        print(f"Data collection failed: {e}")
