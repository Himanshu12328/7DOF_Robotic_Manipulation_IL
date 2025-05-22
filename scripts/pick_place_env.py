import gym
import numpy as np
import mujoco
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "models", "panda.xml")

class PickPlaceEnv(gym.Env):
    def __init__(self, render_mode=False):
        assert os.path.exists(MODEL_PATH), f"Model not found at {MODEL_PATH}"
        self.model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.viewer = None
        if self.render_mode:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.action_space = gym.spaces.Box(
            low=self.model.jnt_range[:7, 0],
            high=self.model.jnt_range[:7, 1],
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "object_initial_pose")
        if key_id != -1:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

        return self._get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:7] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, done = self._compute_reward_done(obs)
        self.step_count += 1

        return obs, reward, done, {}

    def render(self):
        if self.viewer:
            self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()

    def _get_obs(self):
        qpos = self.data.qpos[:7]
        obj_pos = self._get_body_pos("target_object")
        ee_pos = self._get_body_pos("hand")
        rel = obj_pos - ee_pos
        return np.concatenate([qpos, rel]).astype(np.float32)

    def _compute_reward_done(self, obs):
        rel = obs[7:]
        distance = np.linalg.norm(rel)
        reward = -distance

        done = self.step_count >= self.max_steps
        return reward, done

    def _get_body_pos(self, name):
        id_ = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[id_].copy() if id_ != -1 else np.zeros(3)
