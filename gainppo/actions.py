"""Codec owns geometric interpolation; these are only rate selectors."""
import numpy as np
from gymnasium import spaces

GAIN_TABLE = tuple((i // 2, 0.5 if i % 2 else 0.0) for i in range(11))


def action_space(shared_gain=False):
    return spaces.Discrete(11) if shared_gain else spaces.MultiDiscrete([11, 11])


def parse_action(action, shared_gain=False):
    a = np.asarray(action)
    if a.size != (1 if shared_gain else 2) or a.dtype.kind not in 'iu' or a.dtype.kind == 'b':
        raise ValueError('Expected integer gain index(es), one shared or two independent')
    indices = a.reshape(-1).tolist()
    if any(i < 0 or i > 10 for i in indices):
        raise ValueError('Gain indices must be in [0,10]')
    return tuple(indices * 2 if shared_gain else indices)


def rates(action, shared_gain=False):
    return tuple(GAIN_TABLE[i] for i in parse_action(action, shared_gain))
