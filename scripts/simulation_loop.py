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

class BCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 128),
            nn.ReLU(),
            nn.Linear(128, 7)
        )

    def forward(self, x):
        return self.net(x)

model = BCModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

simulation_steps = 500  # Increase steps for longer visualization
blend_factor = 0.1
sleep_time = 0.02  # Controls visualization speed

GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0

def control_gripper(close=True):
    ctrl_value = 255 if close else 0
    gripper_actuator_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8")
    mj_data.ctrl[gripper_actuator_idx] = ctrl_value

def get_object_position():
    obj_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "target_object")
    return mj_data.xpos[obj_body_id]

initial_obj_pos = get_object_position().copy()

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():  # Keeps the viewer open until you manually close it
        for step in range(simulation_steps):
            current_state = mj_data.qpos[:7].copy()

            state_tensor = torch.tensor(current_state.reshape(1, 7), dtype=torch.float32)
            predicted_action = model(state_tensor).detach().numpy().flatten()
            predicted_action = np.clip(predicted_action, -1.0, 1.0)

            new_state = current_state + blend_factor * (predicted_action - current_state)
            mj_data.qpos[:7] = new_state
            mj_data.qvel[:7] = 0

            # Simple Gripper Logic
            if step < simulation_steps // 3:
                control_gripper(close=False)
            else:
                control_gripper(close=True)

            mujoco.mj_step(mj_model, mj_data)

            viewer.sync()
            time.sleep(sleep_time)

        break  # Exit after completing simulation steps

final_obj_pos = get_object_position()
movement_distance = np.linalg.norm(final_obj_pos - initial_obj_pos)
print(f"\nObject Movement Distance: {movement_distance:.4f} meters")

if movement_distance > 0.05:
    print("✅ Task Success: Object was moved!")
else:
    print("❌ Task Failed: Object was not moved.")

print("Simulation completed!")
