import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import os
import time

# Dynamically set paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "bc_model.pth")
XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml")  # Example for a 7-DOF arm

# Safety Checks
if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"Mujoco XML model not found at: {XML_MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained policy model not found at: {MODEL_PATH}")

# Define Behavioral Cloning Model
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

# Load Trained Policy
model = BCModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Initialize Mujoco Simulation
mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
mj_data = mujoco.MjData(mj_model)

# Simulation Parameters
simulation_steps = 500
blend_factor = 0.1  # Controls smoothness of movement
sleep_time = 0.02   # 50 FPS control loop

# Launch Viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    for step in range(simulation_steps):
        # 1. Get Current Joint Positions (First 7 joints for 7-DOF arm)
        current_state = mj_data.qpos[:7].copy()

        # 2. Predict Next Action Using the Policy
        state_tensor = torch.tensor(current_state.reshape(1, 7), dtype=torch.float32)
        predicted_action = model(state_tensor).detach().numpy().flatten()

        # 3. Add Artificial Noise to Visualize Movement (Debugging Purposes Only)
        predicted_action += 0.3 * np.random.randn(7)
        predicted_action = np.clip(predicted_action, -1.0, 1.0)  # Ensure valid range

        # 4. Smoothly Blend Between Current State and Predicted Action
        new_state = current_state + blend_factor * (predicted_action - current_state)
        mj_data.qpos[:7] = new_state
        mj_data.qvel[:7] = 0  # Zero velocities to keep control stable

        # 5. Step Simulation
        mujoco.mj_step(mj_model, mj_data)

        # Debug Output
        print(f"Step: {step}")
        print(f"Current State (qpos): {current_state}")
        print(f"Predicted Action: {predicted_action}\n")

        # 6. Sleep to Control Simulation Speed and Update Viewer
        viewer.sync()
        time.sleep(sleep_time)

        if not viewer.is_running():
            break

print("Simulation completed!")
