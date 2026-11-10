"""Synthetic CPU fixture. Its bytes/rate/success are NOT real Codec/EComp results."""
import numpy as np
from .env import GainEnv


class MockCodec:
    def reset(self, episode_id):
        self.episode_id, self.frame = episode_id, 0

    def process(self, main, wrist, main_rate, wrist_rate):
        # Deliberately nonmonotone numbering: fixture never establishes real RD.
        sizes = [20 + 7*((rate[0]*2+int(rate[1]*2)+3)%11) for rate in (main_rate,wrist_rate)]
        bits = (24+sum(sizes))*8
        pixels = sum(int(np.prod(x.shape[:2])) for x in (main,wrist))
        info = dict(bits=bits, pixels=pixels, actual_bpp=bits/pixels,
            view_bpp=dict(main=sizes[0]*8/main.shape[0]/main.shape[1], wrist=sizes[1]*8/wrist.shape[0]/wrist.shape[1]),
            psnr=dict(main=40., wrist=40.), bpp_type='synthetic_mock',
            episode_id=self.episode_id, frame_index=self.frame)
        self.frame += 1
        return main.astype(np.float32)/255, wrist.astype(np.float32)/255, info


class MockBackend:
    def __init__(self, horizon=8):
        self.horizon = horizon

    def reset(self, seed=None):
        if seed is not None or not hasattr(self, 'rng'):
            self.rng = np.random.default_rng(seed)
        self.sim_step = 0
        self.base = self.rng.integers(0,256,(8,8,3),dtype=np.uint8)
        return {}

    def observe(self, raw):
        im = np.roll(self.base, self.sim_step, axis=0)
        return im, np.flip(im,0).copy(), dict(gripper=[self.sim_step*.01], position=[self.sim_step*.02,0,0], orientation=[0,0,0,1])

    def advance(self, raw, main, wrist):
        self.sim_step += 1
        # Signal depends on decoded views; smooth deterministic toy success.
        success = self.sim_step >= self.horizon and bool(main.mean() > .45)
        return {}, success, self.sim_step >= self.horizon and not success, dict(success=success)

    def close(self):
        pass


def make_mock_env(**kwargs):
    return GainEnv(MockBackend(), MockCodec(), **kwargs)
