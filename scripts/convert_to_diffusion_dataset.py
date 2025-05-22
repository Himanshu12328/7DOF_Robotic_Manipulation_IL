import h5py
import numpy as np
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_FILE = os.path.join(BASE_DIR, "data", "ik_demos.hdf5")
SAVE_DIR = os.path.join(BASE_DIR, "diffusion_dataset")
os.makedirs(SAVE_DIR, exist_ok=True)

dataset = {
    "observations": [],
    "actions": [],
}

with h5py.File(DATA_FILE, "r") as f:
    for i in range(50):  # or however many demos you have
        if f.get(f"states_{i}") is None or f.get(f"actions_{i}") is None:
            continue

        states = f[f"states_{i}"][:]
        actions = f[f"actions_{i}"][:]
        dataset["observations"].append(states.tolist())
        dataset["actions"].append(actions.tolist())

np.savez_compressed(os.path.join(SAVE_DIR, "pick_place_data.npz"), **dataset)
print("✅ Converted dataset saved to:", SAVE_DIR)
