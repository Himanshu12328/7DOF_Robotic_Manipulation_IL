import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) #
MODEL_DIR = os.path.join(BASE_DIR, "models") #
MODEL_PATH = os.path.join(MODEL_DIR, "bc_model.pth") #
XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml") #

if not os.path.exists(XML_MODEL_PATH): #
    raise FileNotFoundError(f"Mujoco XML model not found at: {XML_MODEL_PATH}") #

if not os.path.exists(MODEL_PATH): #
    raise FileNotFoundError(f"Trained policy model not found at: {MODEL_PATH}") #

# Using the model definition from your uploaded simulation_loop.py
class AdvancedBCModel(nn.Module): #
    def __init__(self): #
        super().__init__() #
        self.net = nn.Sequential( #
            nn.Linear(10, 512), #
            nn.BatchNorm1d(512), #
            nn.GELU(), #
            nn.Dropout(0.3), #
            nn.Linear(512, 512), #
            nn.BatchNorm1d(512), #
            nn.GELU(), #
            nn.Dropout(0.3), #
            nn.Linear(512, 256), #
            nn.BatchNorm1d(256), #
            nn.GELU(), #
            nn.Linear(256, 128), #
            nn.GELU(), #
            nn.Linear(128, 7) #
        )

    def forward(self, x): #
        return self.net(x) #

model = AdvancedBCModel() #
model.load_state_dict(torch.load(MODEL_PATH)) #
model.eval() #

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH) #
mj_data = mujoco.MjData(mj_model) #

# Constants from your script
ACTION_DIM = 7 # For arm joints
STATE_DIM = 10 # 7 arm qpos + 3 relative_pos

simulation_steps_per_episode = 1500 # Your variable name was simulation_steps #
blend_factor = 0.3 #
sleep_time = 0.015 #

GRIPPER_CLOSE_DIST_THRESHOLD = 0.045 # Your variable name was GRIPPER_CLOSE_DIST #
GRIPPER_HYSTERESIS = 0.01 #

# Gripper actuator values
GRIPPER_OPEN_CTRL = 0
GRIPPER_CLOSE_CTRL = 255

def control_gripper_actuator(close_gripper=True): # Renamed for consistency with my suggestions
    ctrl_value = GRIPPER_CLOSE_CTRL if close_gripper else GRIPPER_OPEN_CTRL #
    gripper_actuator_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8") #
    if gripper_actuator_idx != -1:
        mj_data.ctrl[gripper_actuator_idx] = ctrl_value #

def get_body_position_sim(body_name):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1: return None
    return mj_data.xpos[body_id]

def get_object_position_sim():
    return get_body_position_sim("target_object") #

def get_end_effector_position_sim():
    return get_body_position_sim("hand") #


def run_simulation():
    gripper_is_closed_heuristic = False  # Hysteresis state, your variable name was gripper_closed #

    mujoco.mj_resetData(mj_model, mj_data)
    try:
        # Ensure the keyframe ID is correctly obtained. `mj_model.keyframe()` returns a MjuiKeyframe object.
        key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "object_initial_pose")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(mj_model, mj_data, key_id)
        else:
            raise ValueError("Keyframe 'object_initial_pose' not found.")
    except Exception as e:
        print(f"Could not reset to keyframe 'object_initial_pose': {e}. Using manual reset.")
        obj_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_free_joint")
        if obj_jnt_id != -1:
            mj_data.qpos[obj_jnt_id:obj_jnt_id+3] = [0.4, 0.0, 0.025]
            mj_data.qpos[obj_jnt_id+3:obj_jnt_id+7] = [1,0,0,0]
    
    # Set arm to home position from "home" keyframe
    home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_key_id != -1:
        # Only apply the arm part of the home qpos after object is set, if needed
        # For now, object_initial_pose keyframe already sets the arm to home.
        # If you want a different arm pose, set mj_data.qpos[:ACTION_DIM] here.
        # mj_data.qpos[:ACTION_DIM] = mj_model.key_qpos[home_key_id*mj_model.nq : home_key_id*mj_model.nq + ACTION_DIM]
        pass # Assuming 'object_initial_pose' sets arm correctly for start

    mujoco.mj_forward(mj_model, mj_data)
    
    initial_obj_pos_check = get_object_position_sim() #
    if initial_obj_pos_check is None:
        print("Error: Could not get initial object position after reset.")
        return
    initial_obj_pos = initial_obj_pos_check.copy()
    print(f"Initial object position: {initial_obj_pos}")


    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer: #
        viewer.cam.lookat = [0.5, 0, 0.3]
        viewer.cam.distance = 2.5
        viewer.cam.elevation = -25
        viewer.cam.azimuth = 135

        while viewer.is_running(): #
            for step in range(simulation_steps_per_episode): #
                if not viewer.is_running(): break #

                current_qpos_arm = mj_data.qpos[:ACTION_DIM].copy() #
                ee_pos = get_end_effector_position_sim() #
                obj_pos = get_object_position_sim() #

                if ee_pos is None or obj_pos is None:
                    print("Error: Could not get EE or Object position. Skipping step.")
                    time.sleep(sleep_time) #
                    continue
                
                relative_pos_ee_obj = obj_pos - ee_pos #

                state_input_np = np.concatenate([current_qpos_arm, relative_pos_ee_obj]) #
                if state_input_np.shape[0] != STATE_DIM:
                    print(f"Error: State input dimension mismatch. Expected {STATE_DIM}, got {state_input_np.shape[0]}")
                    break
                
                state_tensor = torch.tensor(state_input_np.reshape(1, STATE_DIM), dtype=torch.float32) #

                with torch.no_grad():
                    predicted_target_qpos_arm = model(state_tensor).numpy().flatten() #

                # --- CRITICAL FIX: REMOVE INCORRECT CLIPPING ---
                # Original line from your file: predicted_action = np.clip(predicted_action, -1.0, 1.0) #
                # This was wrong. The model outputs actual joint angles.
                # Correct clipping is to actual joint limits:
                joint_ranges = mj_model.jnt_range[:ACTION_DIM] 
                clipped_predicted_target_qpos_arm = np.clip(predicted_target_qpos_arm, joint_ranges[:, 0], joint_ranges[:, 1])
                
                control_target_qpos = current_qpos_arm + blend_factor * (clipped_predicted_target_qpos_arm - current_qpos_arm) #
                mj_data.ctrl[:ACTION_DIM] = control_target_qpos

                distance_to_obj = np.linalg.norm(relative_pos_ee_obj) #
                ee_z_close_enough_to_obj_z = abs(ee_pos[2] - obj_pos[2]) < 0.05 

                if not gripper_is_closed_heuristic and distance_to_obj < GRIPPER_CLOSE_DIST_THRESHOLD and ee_z_close_enough_to_obj_z: #
                    control_gripper_actuator(close_gripper=True) #
                    gripper_is_closed_heuristic = True #
                elif gripper_is_closed_heuristic and distance_to_obj > GRIPPER_CLOSE_DIST_THRESHOLD + GRIPPER_HYSTERESIS: #
                    control_gripper_actuator(close_gripper=False) #
                    gripper_is_closed_heuristic = False #

                mujoco.mj_step(mj_model, mj_data) #
                viewer.sync() #
                if sleep_time > 0: #
                    time.sleep(sleep_time) #
            
            print("Episode finished.") #
            break #

    final_obj_pos_check = get_object_position_sim() #
    if final_obj_pos_check is not None and initial_obj_pos is not None:
        final_obj_pos = final_obj_pos_check
        movement_distance_xy = np.linalg.norm(final_obj_pos[:2] - initial_obj_pos[:2]) 
        movement_distance_xyz = np.linalg.norm(final_obj_pos - initial_obj_pos) #
        print(f"\nObject Movement Distance (XY plane): {movement_distance_xy:.4f} meters")
        print(f"Object Movement Distance (XYZ): {movement_distance_xyz:.4f} meters")

        if movement_distance_xy > 0.05: 
            print("✅ Task Success: Object was moved significantly!") #
        else:
            print("❌ Task Failed: Object was not moved significantly.") #
    else:
        print("Could not determine object movement.")

    print("Simulation completed!") #

if __name__ == "__main__":
    run_simulation() # 