"""Ordered Box(6); raw HWC RGB Pearson and ElegantVLA motion differences."""
from collections import deque
import numpy as np
from gymnasium import spaces

FEATURES = ('main_pearson', 'wrist_pearson', 'gripper_speed',
            'eef_translation_speed', 'eef_rotation_speed', 'step_progress')


def observation_space():
    return spaces.Box(np.array([-1, -1, 0, 0, 0, 0], np.float32),
                      np.array([1, 1, np.inf, np.inf, np.inf, 1], np.float32))


def rgb_float(image):
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[-1] != 3 or min(a.shape[:2]) < 1:
        raise ValueError('Expected nonempty HWC RGB')
    if a.dtype == np.uint8:
        return a.astype(np.float64) / 255.0
    a = a.astype(np.float64)
    if np.isfinite(a).all() and (a.min() < 0 or a.max() > 1):
        raise ValueError('Float RGB must be in [0,1]')
    return a


def pearson(current, previous):
    """Flatten HWC RGB in C order; missing/constant/NaN/size change -> (0,False)."""
    a = rgb_float(current)
    if previous is None:
        return 0.0, False
    b = rgb_float(previous)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return 0.0, False
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a-a.mean(), b-b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return (float(np.clip(np.dot(a, b)/den, -1, 1)), True) if den > 1e-12 else (0.0, False)


class FeatureBuilder:
    def __init__(self, window=4):
        if window < 1:
            raise ValueError('window must be positive')
        self.window = window
        self.reset()

    def reset(self):
        self.images = [None, None]
        self.histories = [deque(maxlen=self.window+1) for _ in range(3)]

    def build(self, main, wrist, state, episode_step, max_episode_steps):
        if max_episode_steps <= 0:
            raise ValueError('max_episode_steps must be positive')
        corr, validity = [], {}
        for i, im in enumerate((main, wrist)):
            value, valid = pearson(im, self.images[i])
            corr.append(value)
            validity[FEATURES[i]] = valid
            self.images[i] = np.asarray(im).copy()
        speeds = []
        for i, key in enumerate(('gripper', 'position', 'orientation')):
            raw = state.get(key)
            a = None if raw is None else np.asarray(raw, dtype=np.float32).reshape(-1)
            valid = a is not None and np.isfinite(a).all() and a.size in ((1,) if i == 0 else (3,) if i == 1 else (3, 4))
            h = self.histories[i]
            if not valid or (h and a.shape != h[-1].shape):
                h.clear()
            if valid:
                h.append(a.copy())
            ready = len(h) == self.window+1
            speed = 0.0
            if ready:
                start, end = h[0], h[-1]
                if i == 2 and a.size == 4:
                    norms = np.linalg.norm(start)*np.linalg.norm(end)
                    ready = norms > 1e-8
                    if ready:
                        speed = 2*np.arccos(np.clip(abs(np.dot(start, end)/norms), -1, 1))/self.window
                else:
                    speed = np.linalg.norm(end-start)/self.window
            speeds.append(float(speed))
            validity[FEATURES[i+2]] = bool(ready)
        obs = np.asarray([*corr, *speeds, np.clip(episode_step/max_episode_steps, 0, 1)], np.float32)
        return obs, {'feature_validity': validity, 'motion_missing_or_warming': not all(validity[k] for k in FEATURES[2:5])}


def ecomp_state(obs):
    """Robosuite xyzw quaternion and first gripper coordinate, no unit conversion."""
    g = obs.get('robot0_gripper_qpos')
    return dict(gripper=None if g is None else np.asarray(g).reshape(-1)[:1],
                position=obs.get('robot0_eef_pos'), orientation=obs.get('robot0_eef_quat'))
