import mujoco
import mujoco.viewer
import numpy as np
import h5py
import os
import time

# Dynamic Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml")
DATA_FILE = os.path.join(DATA_DIR, "collected_data.hdf5")

if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"XML model not found at: {XML_MODEL_PATH}")

# Simulation Parameters
NUM_DEMONSTRATIONS = 10
STEPS_PER_PHASE = 50  # Steps for each phase (approach, grasp, lift)
TOTAL_STEPS = STEPS_PER_PHASE * 3

# Load Mujoco Model
mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

def move_towards_target(current_state, target_pos, blend_factor=0.05):
    """
    Simple controller to move joints slightly towards target joint positions.
    """
    return current_state + blend_factor * (target_pos - current_state)

def scripted_demo(index):
    """
    Simulate one scripted pick-and-place demonstration.
    """
    states = []
    actions = []

    # Reset Environment
    mujoco.mj_resetData(mj_model, mj_data)

    # Target Object Position (from XML)
    target_obj_pos = np.array([0.5, 0, 0.05])

    # Predefined Fake Trajectories for Pick-and-Place Phases
    # Approach Phase: Move arm towards object
    target_approach = np.random.uniform(-0.5, 0.5, 7)

    # Grasp Phase: Assume a fixed grasp joint config
    target_grasp = np.random.uniform(-0.2, 0.2, 7)

    # Lift Phase: Slightly increase one joint to simulate lifting
    target_lift = target_grasp + np.array([0.1] * 7)

    phase_targets = [target_approach, target_grasp, target_lift]

    for phase_idx, target in enumerate(phase_targets):
        for step in range(STEPS_PER_PHASE):
            current_state = mj_data.qpos[:7].copy()

            # Simple joint-space movement controller
            next_state = move_towards_target(current_state, target)
            mj_data.qpos[:7] = next_state
            mj_data.qvel[:7] = 0

            mujoco.mj_step(mj_model, mj_data)

            # Record Data
            states.append(current_state)
            actions.append(target)

    return np.array(states), np.array(actions)

def collect_scripted_demos():
    """
    Collect multiple demonstrations and save them in HDF5 format.
    """
    with h5py.File(DATA_FILE, 'w') as f:
        for demo_idx in range(NUM_DEMONSTRATIONS):
            states, actions = scripted_demo(demo_idx)
            f.create_dataset(f'states_{demo_idx}', data=states)
            f.create_dataset(f'actions_{demo_idx}', data=actions)
            print(f"Collected demonstration {demo_idx + 1}/{NUM_DEMONSTRATIONS}")

    print(f"Scripted demonstrations saved at: {DATA_FILE}")

if __name__ == "__main__":
    collect_scripted_demos()
