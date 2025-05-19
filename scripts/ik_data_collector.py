import mujoco
import mujoco.viewer
import numpy as np
import h5py
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml")
DATA_FILE = os.path.join(DATA_DIR, "ik_demos.hdf5")

if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"XML model not found at: {XML_MODEL_PATH}")

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

NUM_DEMOS = 50
STEPS_PER_DEMO = 150

def get_object_position():
    obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    return mj_data.xpos[obj_body_id]

def get_end_effector_position():
    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return mj_data.xpos[ee_body_id]

def scripted_demo():
    states = []
    actions = []

    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    obj_pos = get_object_position()
    mocap_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mocap_ee")

    for step in range(STEPS_PER_DEMO):
        ee_pos = get_end_effector_position()
        direction = obj_pos - ee_pos
        direction /= np.linalg.norm(direction) + 1e-6  # Normalize

        # Set mocap target position directly toward the object
        target_pos = ee_pos + 0.01 * direction
        mj_data.mocap_pos[mocap_body_id - mj_model.nbody + mj_model.nmocap] = target_pos

        mujoco.mj_forward(mj_model, mj_data)

        current_joint_positions = mj_data.qpos[:7].copy()
        relative_pos = obj_pos - ee_pos
        combined_state = np.concatenate([current_joint_positions, relative_pos])

        states.append(combined_state)
        actions.append(current_joint_positions)

        mujoco.mj_step(mj_model, mj_data)

    return np.array(states), np.array(actions)

def collect_ik_demos():
    with h5py.File(DATA_FILE, 'w') as f:
        for demo_idx in range(NUM_DEMOS):
            states, actions = scripted_demo()
            f.create_dataset(f'states_{demo_idx}', data=states)
            f.create_dataset(f'actions_{demo_idx}', data=actions)
            print(f"Collected IK demonstration {demo_idx + 1}/{NUM_DEMOS}")

    print(f"IK demonstrations saved at: {DATA_FILE}")

if __name__ == "__main__":
    collect_ik_demos()
