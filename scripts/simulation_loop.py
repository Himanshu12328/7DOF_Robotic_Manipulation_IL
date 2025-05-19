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
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)

model = AdvancedBCModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

simulation_steps = 1500
blend_factor = 0.3
sleep_time = 0.015

GRIPPER_CLOSE_DIST = 0.05
GRIPPER_HYSTERESIS = 0.01  # Prevent flickering

def control_gripper(close=True):
    ctrl_value = 255 if close else 0
    gripper_actuator_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    mj_data.ctrl[gripper_actuator_idx] = ctrl_value

def get_object_position():
    obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    return mj_data.xpos[obj_body_id]

def get_end_effector_position():
    ee_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    return mj_data.xpos[ee_body_id]

initial_obj_pos = get_object_position().copy()
gripper_closed = False  # Hysteresis state

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        for step in range(simulation_steps):
            current_state = mj_data.qpos[:7].copy()
            ee_pos = get_end_effector_position()
            obj_pos = get_object_position()
            relative_pos = obj_pos - ee_pos

            # Prepare input for model
            state_input = np.concatenate([current_state, relative_pos])
            state_tensor = torch.tensor(state_input.reshape(1, 10), dtype=torch.float32)

            predicted_action = model(state_tensor).detach().numpy().flatten()
            predicted_action = np.clip(predicted_action, -1.0, 1.0)

            # Smoothly apply joint movements
            new_state = current_state + blend_factor * (predicted_action - current_state)
            mj_data.qpos[:7] = new_state
            mj_data.qvel[:7] = 0

            # Intelligent Gripper Control with Hysteresis
            distance_to_obj = np.linalg.norm(relative_pos)
            if not gripper_closed and distance_to_obj < GRIPPER_CLOSE_DIST:
                control_gripper(close=True)
                gripper_closed = True
            elif gripper_closed and distance_to_obj > GRIPPER_CLOSE_DIST + GRIPPER_HYSTERESIS:
                control_gripper(close=False)
                gripper_closed = False

            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()
            time.sleep(sleep_time)

        break

final_obj_pos = get_object_position()
movement_distance = np.linalg.norm(final_obj_pos - initial_obj_pos)
print(f"\nObject Movement Distance: {movement_distance:.4f} meters")

if movement_distance > 0.05:
    print("✅ Task Success: Object was moved!")
else:
    print("❌ Task Failed: Object was not moved.")

print("Simulation completed!")
