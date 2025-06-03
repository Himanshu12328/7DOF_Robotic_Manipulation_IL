import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bc_model.pth")
XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml")

if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"Mujoco XML model not found at: {XML_MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained policy model not found at: {MODEL_PATH}")

# Using the model definition from your uploaded simulation_loop.py
class AdvancedBCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        return self.net(x)

model = AdvancedBCModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

# Define constants for the simulation
ACTION_DIM = 7 # For arm joints
STATE_DIM = 10 # 7 arm qpos + 3 relative_pos
GRIPPER_ACTUATOR_NAME = "actuator8" # Name of the gripper actuator in the XML
simulation_steps_per_episode = 1500 # Your variable name was simulation_steps
blend_factor = 1
sleep_time = 0.015

def get_body_position(body_name):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in the model.")
    return mj_data.xpos[body_id]

def run_simulation():
    mujoco.mj_resetData(mj_model, mj_data)
    # Reset the simulation data to initial state
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "object_initial_pose")
    if key_id != -1:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
    
    mujoco.mj_forward(mj_model, mj_data)
    
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.lookat = [0.5, 0, 0.3]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        while viewer.is_running():
            for step in range(simulation_steps_per_episode):
                if not viewer.is_running(): 
                    break

                qpos_arm = mj_data.qpos[:ACTION_DIM]  # Get the current arm joint positions
                ee_pos = get_body_position("hand")
                obj_pos = get_body_position("target_object")
                rel = obj_pos - ee_pos  # Relative position of the object to the end effector

                state = np.concatenate([qpos_arm, rel])  # Combine arm joint positions and relative position
                state_tensor = torch.tensor(state.reshape(1, -1), dtype=torch.float32)  # Reshape for model input

                with torch.no_grad():
                    predicted = model(state_tensor).numpy().flatten()  # Get the model's prediction

                predicted_qpos = predicted[:7]  # First 7 values are joint positions
                gripper_control = predicted[7]  # Last value is gripper control
                joint_ranges = mj_model.jnt_range[:ACTION_DIM]  # Get joint limits
                clipped_qpos = np.clip(predicted_qpos, joint_ranges[:, 0], joint_ranges[:, 1])  # Clip to joint limits
                
                # --- Debugging output ---
                print(f"Predicted qpos: {predicted_qpos}")
                print(f"Clipped qpos: {clipped_qpos}")
                print(f"Current qpos: {qpos_arm}")
                print(f"Delta: {clipped_qpos - qpos_arm}")

                # control_target_qpos = qpos_arm + blend_factor * (clipped_qpos - qpos_arm)  # Blend the control target
                mj_data.qpos[:7] = clipped_qpos
                mujoco.mj_forward(mj_model, mj_data)
                gripper_actuator_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR_NAME)
                if gripper_actuator_idx != -1:
                    # Use a threshold for open/close based on BC model output
                    mj_data.ctrl[gripper_actuator_idx] = 225 if gripper_control > 0 else 0

                mujoco.mj_step(mj_model, mj_data)  # Step the simulation
                viewer.sync()
                time.sleep(sleep_time)

            print("Episode finished.")
            break

    print("Simulation completed!") #

if __name__ == "__main__":
    run_simulation() # 