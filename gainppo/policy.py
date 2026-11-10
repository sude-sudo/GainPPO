"""Ordinary SB3 PPO, two categorical branches and one scalar value."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .actions import GAIN_TABLE
from .observation import FEATURES

DEFAULTS = dict(learning_rate=1e-4, gamma=.99, gae_lambda=.95, clip_range=.2,
                n_steps=128, batch_size=128, n_epochs=4, vf_coef=.5, max_grad_norm=.5)


def make_vec(factory, stats=None, training=True):
    vec = DummyVecEnv([factory])
    vec = VecNormalize.load(str(stats), vec) if stats else VecNormalize(vec, norm_obs=True, norm_reward=False)
    vec.training, vec.norm_reward = training, False
    return vec


def build_ppo(env, seed=42, device='cpu', ppo=None, tensorboard_log=None):
    return PPO('MlpPolicy', env, seed=seed, device=device, tensorboard_log=tensorboard_log,
        policy_kwargs=dict(net_arch=dict(pi=[64,64], vf=[64,64]), activation_fn=torch.nn.Tanh, ortho_init=True),
        **(DEFAULTS | (ppo or {})))


def actor_state(policy):
    return {k:v.detach().clone() for k,v in policy.state_dict().items()
            if k.startswith(('mlp_extractor.policy_net.', 'action_net.'))}


def actor_digest(policy):
    digest = hashlib.sha256()
    for k,v in sorted(actor_state(policy).items()):
        digest.update(k.encode()); digest.update(v.cpu().numpy().tobytes())
    return digest.hexdigest()


def initialize_from_bc(model, checkpoint):
    # Newly constructed model keeps fresh critic and optimizer/rollout/progress.
    data, params, _ = load_from_zip_file(str(checkpoint), device=model.device)
    if data['observation_space'] != model.observation_space or data['action_space'] != model.action_space:
        raise ValueError('BC shape/space mismatch')
    state = {k:v for k,v in params['policy'].items()
             if k.startswith(('mlp_extractor.policy_net.', 'action_net.'))}
    result = model.policy.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError('BC actor shape mismatch')
    return actor_digest(model.policy)


class RuntimePolicy:
    def __init__(self, directory, env_factory):
        self.vec = make_vec(env_factory, Path(directory)/'vecnormalize.pkl', training=False)
        self.model = PPO.load(str(Path(directory)/'model.zip'), env=self.vec, device='cpu')

    def predict(self, observation, deterministic=True):
        a = np.asarray(observation, np.float32)
        if a.shape != (6,) or not np.isfinite(a).all():
            raise ValueError('Expected finite ordered six-feature observation')
        return self.model.predict(self.vec.normalize_obs(a.copy()), deterministic=deterministic)[0]

    def close(self):
        self.vec.close()


def load_policy(lambda_id, root='outputs/sweep', env_factory=None):
    from .mock_env import make_mock_env
    return RuntimePolicy(Path(root)/lambda_id, env_factory or make_mock_env)
