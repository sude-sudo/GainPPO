"""One scheduling transition per real Codec/VLA request."""
import gymnasium as gym
import numpy as np
from .actions import action_space, parse_action, rates
from .observation import FeatureBuilder, observation_space


def terminal_reward(success, bits, pixels, lambda_bpp, B_ref, w_s=1.0):
    if not np.isfinite([bits, pixels, lambda_bpp, B_ref, w_s]).all() or pixels <= 0 or bits < 0 or B_ref <= 0 or lambda_bpp < 0:
        raise ValueError('Invalid reward parameters')
    return float(w_s*bool(success) - lambda_bpp*(bits/pixels)/B_ref)


class GainEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, backend, codec, lambda_bpp=0.1, B_ref=1., w_s=1., shared_gain=False, reward_mode='terminal'):
        if reward_mode != 'terminal':
            raise ValueError('Only terminal mean-BPP reward is implemented')
        terminal_reward(False, 0, 1, lambda_bpp, B_ref, w_s)
        self.backend, self.codec = backend, codec
        self.lambda_bpp, self.B_ref, self.w_s = lambda_bpp, B_ref, w_s
        self.shared_gain = shared_gain
        self.action_space, self.observation_space = action_space(shared_gain), observation_space()
        self.features = FeatureBuilder()
        self.episode_id = -1
        self.done = True

    def _observation(self):
        main, wrist, state = self.backend.observe(self.raw)
        obs, info = self.features.build(main, wrist, state, self.backend.sim_step, self.backend.horizon)
        self.current_images = (main, wrist)
        return obs, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.raw = self.backend.reset(seed=seed)
        self.features.reset()
        self.episode_id += 1
        self.codec.reset(self.episode_id)
        self.bits = self.pixels = self.decisions = 0
        self.view_bits = dict(main=0., wrist=0.)
        self.view_pixels = dict(main=0, wrist=0)
        self.psnr_sum = dict(main=0., wrist=0.)
        self.latency_sum = dict(encode_seconds=0., decode_seconds=0., vla_seconds=0.)
        self.done = self.success = False
        self.histogram = np.zeros((2, 11), dtype=int)
        obs, info = self._observation()
        self.current_obs = obs.copy()
        self.current_validity = info["feature_validity"]
        return obs, info

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset before stepping an ended episode')
        source_obs = self.current_obs.copy()
        source_validity = self.current_validity.copy()
        indices = parse_action(action, self.shared_gain)
        main_rate, wrist_rate = rates(action, self.shared_gain)
        main, wrist, info = self.codec.process(*self.current_images, main_rate, wrist_rate)
        for view, image in zip(('main','wrist'), self.current_images):
            n = int(np.prod(image.shape[:2]))
            self.view_bits[view] += info['view_bpp'][view]*n
            self.view_pixels[view] += n
            self.psnr_sum[view] += info['psnr'][view]
        self.bits += info['bits']
        self.pixels += info['pixels']
        self.decisions += 1
        self.histogram[np.arange(2), indices] += 1
        # Backend gets reconstructed RGB only; raw images are never used by VLA.
        self.raw, terminated, truncated, result = self.backend.advance(self.raw, main, wrist)
        for key in self.latency_sum:
            self.latency_sum[key] += float(info.get(key, result.get(key, 0.)))
        self.success |= bool(result['success'])
        self.done = bool(terminated or truncated)
        reward = terminal_reward(self.success, self.bits, self.pixels, self.lambda_bpp, self.B_ref, self.w_s) if self.done else 0.0
        obs, validity = self._observation()  # genuine final state for timeout bootstrap
        info.update(result)
        info.update(validity)
        info.update(observation=source_obs.tolist(), observation_validity=source_validity, action=list(indices), rates=[list(main_rate), list(wrist_rate)],
                    sim_step=self.backend.sim_step, decision_step=self.decisions,
                    reward_source='terminal_episode' if self.done else 'zero_nonterminal',
                    success=self.success, reward=reward, terminated=bool(terminated), truncated=bool(truncated))
        if self.done:
            info['episode_metrics'] = dict(success=self.success, avg_bpp=self.bits/self.pixels,
                bits=self.bits, pixels=self.pixels, decisions=self.decisions,
                sim_steps=self.backend.sim_step, gain_histogram=self.histogram.tolist(),
                bpp_type=info['bpp_type'],
                view_bpp={v:self.view_bits[v]/self.view_pixels[v] for v in self.view_bits},
                psnr={v:self.psnr_sum[v]/self.decisions for v in self.psnr_sum},
                latency_seconds=self.latency_sum.copy())
        self.current_obs = obs.copy()
        self.current_validity = validity["feature_validity"]
        return obs, reward, bool(terminated), bool(truncated), info

    def close(self):
        self.backend.close()
