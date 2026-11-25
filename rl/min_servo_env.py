"""
Gymnasium environment: reach a 3D target with the myCobot 280 while using as few
servos as possible.

Each step the policy outputs a small change for all 6 joints (scaled to
+/- dq_max rad). The reward is

    r = -distance(tip, target)
        - servo_penalty * (# servos that start moving for the first time this step)   <- L0 cost
        - action_penalty * sum|action|                                               <- L1 sparsity push
        + success_bonus when the tip is within `success_tol`

The observation includes which servos have already moved, so the policy can
learn to keep reusing the same ones.

Targets are sampled by moving 3 random servos from the start pose, so every
target is reachable with at most 3 servos (the same minimum the exact solver finds).
"""
import pathlib
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_XML = ROOT / "models" / "mycobot_280_m5.xml"


class MinServoReachEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, xml_path=DEFAULT_XML, max_steps=100, dq_max=0.05, servo_penalty=0.5,
                 action_penalty=0.01, success_tol=0.005, success_bonus=10.0,
                 random_start=True, render_mode=None):
        self.m = mujoco.MjModel.from_xml_path(str(xml_path))
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("tip").id
        ids = [self.m.joint(f"joint{i}").id for i in range(1, 7)]
        self.qadr = self.m.jnt_qposadr[ids]
        self.lo, self.hi = self.m.jnt_range[ids].T.copy()
        self.max_steps, self.dq_max = max_steps, dq_max
        self.servo_penalty, self.action_penalty = servo_penalty, action_penalty
        self.success_tol, self.success_bonus = success_tol, success_bonus
        self.random_start, self.eps = random_start, 1e-3
        self.render_mode, self._renderer = render_mode, None
        self.action_space = spaces.Box(-1.0, 1.0, (6,), np.float32)
        # q (6) + moved flags (6) + tip (3) + target - tip (3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (18,), np.float32)

    # ---- helpers -------------------------------------------------------
    def _set(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy()

    def _obs(self):
        tip = self.d.site_xpos[self.site]
        moved = (np.abs(self.q - self.q0) > self.eps).astype(np.float64)
        return np.concatenate([self.q, moved, tip, self.target - tip]).astype(np.float32)

    def _sample_target(self):
        q = self.q0.copy()
        idx = self.np_random.choice(5, 3, replace=False)       # J6 never moves the tip
        q[idx] = self.np_random.uniform(self.lo[idx], self.hi[idx]) * 0.8
        tip = self._set(q)
        if tip[2] < 0.02:                                          # keep targets above the table
            return self._sample_target()
        return tip

    # ---- gym API -------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q0 = (self.np_random.uniform(self.lo, self.hi) * 0.4 if self.random_start
                   else np.zeros(6))
        self.target = (np.asarray(options["target"], float) if options and "target" in options
                       else self._sample_target())
        if options and "q0" in options:
            self.q0 = np.asarray(options["q0"], float)
        self.q = self.q0.copy()
        self._set(self.q)
        self.d.mocap_pos[0] = self.target
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        moved_before = np.abs(self.q - self.q0) > self.eps
        self.q = np.clip(self.q + action * self.dq_max, self.lo, self.hi)
        tip = self._set(self.q)
        self.t += 1
        dist = float(np.linalg.norm(self.target - tip))
        moved_now = np.abs(self.q - self.q0) > self.eps

        reward = -dist
        reward -= self.servo_penalty * float((moved_now & ~moved_before).sum())
        reward -= self.action_penalty * float(np.abs(action).sum())
        success = dist < self.success_tol
        if success:
            reward += self.success_bonus
        info = {"distance": dist, "servos_used": int(moved_now.sum()), "is_success": success}
        return self._obs(), reward, success, self.t >= self.max_steps, info

    def render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, 480, 640)
        mujoco.mj_forward(self.m, self.d)
        self._renderer.update_scene(self.d)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
