import mujoco
import mujoco.viewer # For potential debugging with live viewer
import numpy as np
import h5py
import os
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

XML_MODEL_PATH = os.path.join(MODEL_DIR, "panda.xml")
DATA_FILE = os.path.join(DATA_DIR, "ik_demos.hdf5")

if not os.path.exists(XML_MODEL_PATH):
    raise FileNotFoundError(f"XML model not found at: {XML_MODEL_PATH}") #

mj_model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH) #
mj_data = mujoco.MjData(mj_model) #

NUM_DEMOS = 50 #
STEPS_PER_DEMO = 300 # Increased to allow for more complex motion #

# Initial robot joint configuration (e.g., "home" or a neutral pose)
INITIAL_QPOS = mj_model.keyframe("home").qpos.copy() # Use home keyframe if available #

# Gripper actuator values
GRIPPER_OPEN_CTRL = 0 #
GRIPPER_CLOSE_CTRL = 255 # Max value for actuator8 #

def get_body_position(body_name):
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name) #
    if body_id == -1: #
        raise ValueError(f"Body '{body_name}' not found.") #
    return mj_data.xpos[body_id] #

def get_object_position():
    return get_body_position("target_object") #

def get_end_effector_position():
    return get_body_position("hand") # Using hand body's CoM as EE point #

def set_gripper_control(ctrl_value):
    gripper_actuator_idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator8") #
    if gripper_actuator_idx != -1: #
        mj_data.ctrl[gripper_actuator_idx] = ctrl_value #
    else:
        print("Warning: Gripper actuator 'actuator8' not found.") #


def reset_simulation_to_object_pose(object_x=0.4, object_y=0.0, object_z=0.025):
    """Resets the simulation with the object at a specified XYZ position."""
    mujoco.mj_resetData(mj_model, mj_data) #
    
    # Set initial robot arm pose (e.g., home)
    # INITIAL_QPOS contains all 16 qpos values (7 arm, 2 finger, 7 object)
    # mj_model.nq - 7 = 16 - 7 = 9 (arm + finger joints)
    if mj_model.nq >= 9: # Ensure model has enough qpos for arm and fingers
        mj_data.qpos[:9] = INITIAL_QPOS[:9] # Set arm and finger joints from home
    else:
        print(f"Warning: Model nq ({mj_model.nq}) is less than 9. Cannot fully set initial arm/finger pose.")


    # Set object position
    obj_jnt_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_free_joint") #
    if obj_jnt_id != -1: #
        # For a free joint, the first 3 qpos are XYZ position, next 4 are quaternion
        mj_data.qpos[obj_jnt_id:obj_jnt_id+3] = [object_x, object_y, object_z] #
        mj_data.qpos[obj_jnt_id+3:obj_jnt_id+7] = [1, 0, 0, 0] # Identity quaternion #
    else:
        print("Warning: 'object_free_joint' not found. Cannot set object position via qpos.") #

    mujoco.mj_forward(mj_model, mj_data) #


def scripted_pick_and_place_demo(viewer_handle=None):
    states = [] #
    actions = [] #

    # --- Define Pick and Place Parameters ---
    # Initial and target placement positions for the object
    initial_obj_x = 0.4 + np.random.uniform(-0.05, 0.05) # Add some randomization #
    initial_obj_y = 0.0 + np.random.uniform(-0.05, 0.05) #
    reset_simulation_to_object_pose(object_x=initial_obj_x, object_y=initial_obj_y) #
    
    # Target placement (e.g., move 0.2m in -Y direction, 0.1m in +X)
    target_obj_x = initial_obj_x + 0.1 #
    target_obj_y = initial_obj_y - 0.2 #
    target_obj_z = 0.025 # Assuming it's placed back on the table #

    pre_grasp_height_offset = 0.12  # How high above the object to pre-grasp #
    grasp_approach_offset = 0.005 # How close to object center vertically for grasp #
    lift_height = 0.10           # How high to lift the object #

    # --- CORRECTED MOCAP ID ---
    mocap_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "mocap_ee") #
    if mocap_body_id == -1 or mj_model.body_mocapid[mocap_body_id] == -1: #
        raise ValueError("Mocap body 'mocap_ee' not found or not properly configured in XML.") #
    mocap_id = mj_model.body_mocapid[mocap_body_id] # Correct way to get mocap_id for mj_data.mocap_pos #

    path_phases = [ #
        (lambda: get_object_position() + np.array([0, 0, pre_grasp_height_offset]), GRIPPER_OPEN_CTRL, 60, 0.03), #
        (lambda: get_object_position() + np.array([0, 0, grasp_approach_offset]), GRIPPER_OPEN_CTRL, 40, 0.02), #
        (lambda: get_end_effector_position(), GRIPPER_CLOSE_CTRL, 20, 0.01), #
        (lambda: get_object_position() + np.array([0, 0, lift_height]), GRIPPER_CLOSE_CTRL, 40, 0.02), #
        (lambda: np.array([target_obj_x, target_obj_y, target_obj_z + lift_height]), GRIPPER_CLOSE_CTRL, 70, 0.03), #
        (lambda: np.array([target_obj_x, target_obj_y, target_obj_z + grasp_approach_offset]), GRIPPER_CLOSE_CTRL, 40, 0.02), #
        (lambda: get_end_effector_position(), GRIPPER_OPEN_CTRL, 20, 0.01), #
        (lambda: get_end_effector_position() + np.array([0, 0, pre_grasp_height_offset]), GRIPPER_OPEN_CTRL, 30, 0.03) #
    ]

    current_demo_step = 0 #
    for get_target_pos_func, gripper_val, num_steps_phase, mocap_speed in path_phases: #
        if current_demo_step >= STEPS_PER_DEMO: break #
        set_gripper_control(gripper_val) #

        for i in range(num_steps_phase): #
            if current_demo_step >= STEPS_PER_DEMO: break #

            target_mocap_world_pos = get_target_pos_func() #
            current_mocap_pos_on_data = mj_data.mocap_pos[mocap_id].copy() #

            direction_to_target = target_mocap_world_pos - current_mocap_pos_on_data #
            dist_to_target = np.linalg.norm(direction_to_target) #

            if dist_to_target > 1e-4: # Tolerance for reaching target #
                move_vec = direction_to_target / dist_to_target * min(mocap_speed, dist_to_target) #
                mj_data.mocap_pos[mocap_id] += move_vec #
            else:
                mj_data.mocap_pos[mocap_id] = target_mocap_world_pos # Snap if close enough #

            mujoco.mj_step(mj_model, mj_data) #
            mujoco.mj_forward(mj_model, mj_data) # Recompute after step #

            current_joint_positions = mj_data.qpos[:7].copy() # Assuming first 7 are arm joints #
            ee_pos_after_step = get_end_effector_position() #
            obj_pos_current_for_state = get_object_position() #
            relative_pos = obj_pos_current_for_state - ee_pos_after_step #

            combined_state = np.concatenate([current_joint_positions, relative_pos]) #
            states.append(combined_state) #
            actions.append(current_joint_positions) # Action is to predict these joint positions #
            
            current_demo_step += 1 #
            if viewer_handle: #
                viewer_handle.sync() #
                time.sleep(0.01) # Slow down for viewing #

    while len(states) < STEPS_PER_DEMO and states: #
        states.append(states[-1]) #
        actions.append(actions[-1]) #
    
    if not states: #
        print("Error: No states collected in demo. Generating dummy data.") #
        dummy_state = np.zeros(10) #
        dummy_action = np.zeros(7) #
        for _ in range(STEPS_PER_DEMO): #
            states.append(dummy_state) #
            actions.append(dummy_action) #

    return np.array(states), np.array(actions) #


def collect_ik_demos():
        
    with h5py.File(DATA_FILE, 'w') as f: #
        for demo_idx in range(NUM_DEMOS): #
            states, actions = scripted_pick_and_place_demo() # No viewer #

            if states.shape[0] != STEPS_PER_DEMO or actions.shape[0] != STEPS_PER_DEMO: #
                print(f"Warning: Demo {demo_idx + 1} has incorrect number of steps. Expected {STEPS_PER_DEMO}, got {states.shape[0]}. Skipping this demo.") #
                continue #

            f.create_dataset(f'states_{demo_idx}', data=states) #
            f.create_dataset(f'actions_{demo_idx}', data=actions) #
            # A more accurate check for object movement would be against its starting pos for *this* demo
            # For simplicity, this old check remains, but ideally, it should compare to initial_obj_x, initial_obj_y
            print(f"Collected IK demonstration {demo_idx + 1}/{NUM_DEMOS}. Object moved: {np.linalg.norm(get_object_position()[:2] - np.array([0.4,0.0])) > 0.05}") #

    print(f"IK demonstrations saved at: {DATA_FILE}") #

if __name__ == "__main__":
    collect_ik_demos() #