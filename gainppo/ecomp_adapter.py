"""Compatibility with audited local reproduce.py; no simulator imports at import time."""
import time
import numpy as np
import torch
from .observation import ecomp_state
from .codec_adapter import freeze


class ECompBackend:
    def __init__(self, env_factory, policy, horizon=250, wait_steps=50,
                 prompt='Pick up the object', remote_frozen_confirmed=False):
        self.env_factory, self.policy = env_factory, policy
        self.horizon, self.wait_steps, self.prompt = horizon, wait_steps, prompt
        if isinstance(policy, torch.nn.Module):
            freeze(policy)
        elif not remote_frozen_confirmed:
            raise ValueError('Remote VLA requires explicit frozen deployment confirmation')
        self.env = None

    def reset(self, seed=None):
        # Recreate as baseline seeds numpy before environment construction.
        self.close()
        if seed is not None or not hasattr(self, 'task_rng'):
            self.task_rng = np.random.default_rng(seed)
        self.task_seed = int(seed) if seed is not None else int(self.task_rng.integers(0, 2**31))
        np.random.seed(self.task_seed)
        self.env = self.env_factory()
        self.policy.reset()
        if self.env.action_dim != 7:
            raise ValueError('Audited EComp backend expects UR5e action_dim=7')
        raw = self.env.reset()
        for _ in range(self.wait_steps):
            raw, _, done, _ = self.env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                raise RuntimeError('Terminated during settling')
        self.sim_step = 0
        return raw

    def observe(self, raw):
        images = []
        for key in ('agentview_image', 'robot0_eye_in_hand_image'):
            im = np.asarray(raw[key])
            if im.dtype != np.uint8 or im.shape != (256, 256, 3):
                raise ValueError(f'Baseline requires uint8 256x256 RGB: {key}')
            images.append(np.ascontiguousarray(im[::-1, ::-1]))
        return *images, ecomp_state(raw)

    def advance(self, raw, main_hat, wrist_hat):
        state = np.concatenate((raw['robot0_joint_pos'], [raw['robot0_gripper_qpos'][0]]))
        if state.shape != (7,) or not np.isfinite(state).all():
            raise ValueError('Expected finite UR5e state')
        payload = dict(state=state, prompt=self.prompt,
                       base_rgb=np.rint(main_hat*255).clip(0,255).astype(np.uint8),
                       wrist_rgb=np.rint(wrist_hat*255).clip(0,255).astype(np.uint8))
        t = time.perf_counter()
        with torch.inference_mode():
            response = self.policy.infer(payload)
        elapsed = time.perf_counter()-t
        actions = np.asarray(response['actions'], dtype=np.float64)
        if actions.ndim != 2 or actions.shape[0] < 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError('VLA must return >=2 finite 7D actions')
        same = np.array_equal(actions[0], actions[1])
        terminated = truncated = success = False
        for action in actions[:1 if same else 10]:
            action = action.copy()
            if same:
                action[:3] *= 2
                action[3:6] *= .01
            raw, _, done, _ = self.env.step(action)
            self.sim_step += 1
            success = bool(self.env._check_success())
            truncated = self.sim_step >= self.horizon and not success
            terminated = success or (bool(done) and not truncated)
            if terminated or truncated:
                break
        return raw, terminated, truncated, dict(success=success, vla_seconds=elapsed, task_seed=self.task_seed)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
