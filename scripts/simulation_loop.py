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

# Load the Mujoco model (Ensure you have a valid Mujoco XML model file in your project)
XML_MODEL_PATH = os.path.join(BASE_DIR, "models", "panda.xml")  # Example for a 7-DOF arm

if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"Mujoco XML model not found at: {XML_MODEL_PATH}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Trained policy model not found at: {MODEL_PATH}")

# Define the BCModel (same as training)
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

# Viewer for visualization
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    for step in range(500):  # Run simulation for 500 steps
        # 1. Get Current Robot Joint Positions (First 7 joints for 7-DOF arm)
        current_state = mj_data.qpos[:7].copy()  # Shape: (7,)

        # 2. Predict Next Action from Policy
        state_tensor = torch.tensor(current_state.reshape(1, 7), dtype=torch.float32)
        predicted_action = model(state_tensor).detach().numpy().flatten()  # Shape: (7,)

        # 3. Apply Action: Set new joint positions (this is a simple direct control, you can improve it)
        mj_data.qpos[:7] = predicted_action
        mj_data.qvel[:7] = 0  # Reset velocities to avoid instability

        # 4. Step Simulation Forward
        mujoco.mj_step(mj_model, mj_data)
        
        print(f"Step: {step}")
        print(f"Current State (qpos): {current_state}")
        print(f"Predicted Action: {predicted_action}\n")


        # 5. Slow down for visualization
        time.sleep(0.02)  # 50 FPS

        # Exit if viewer window is closed
        if not viewer.is_running():
            break

print("Simulation completed!")
