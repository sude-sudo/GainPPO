"""Stage 2: one independent PPO per lambda; conservative checkpoint provenance."""
import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from .config import read_config, env_factory
from .policy import build_ppo, make_vec, initialize_from_bc
from .observation import FEATURES
from .actions import GAIN_TABLE


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(config):
    # Output destination is operational, all experimental settings remain locked.
    return {k:v for k,v in config.items() if k not in ('output','resume')}


class MetricsCallback(BaseCallback):
    def __init__(self, output):
        super().__init__()
        self.output=Path(output)
        self.histogram=np.zeros((2,11),int)

    def _on_training_start(self):
        self.stream=(self.output/'steps.jsonl').open('x')

    def _on_step(self):
        for info in self.locals['infos']:
            self.histogram[np.arange(2),info['action']] += 1
            self.stream.write(json.dumps(info,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
            if 'episode_metrics' in info:
                for key in ('avg_bpp','success','sim_steps','decisions'):
                    self.logger.record('episode/'+key,float(info['episode_metrics'][key]))
        for i,view in enumerate(('main','wrist')):
            for a,count in enumerate(self.histogram[i]):
                self.logger.record(f'gain/{view}_{a}',int(count))
        return True

    def _on_rollout_end(self):
        # SB3 records entropy_loss/value_loss/policy_gradient_loss/approx_kl/clip_fraction.
        with torch.no_grad():
            obs=self.model.policy.obs_to_tensor(self.model._last_obs)[0]
            d=self.model.policy.get_distribution(obs)
            branches=d.distribution if isinstance(d.distribution,list) else [d.distribution]
            for i,branch in enumerate(branches):
                self.logger.record(f'distribution/entropy_{i}',float(branch.entropy().mean()))
                for a,p in enumerate(branch.probs.mean(0).cpu().numpy()):
                    self.logger.record(f'distribution/head_{i}_p_{a}',float(p))

    def _on_training_end(self):
        self.stream.close()


def train(config, authorized_real=False, resume=None):
    config=dict(config)
    output=Path(config['output'])
    # Every invocation, including resume, writes a new directory.
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(config.get('cpu_threads',1))
    device=config.get('device','cpu')
    if device != 'cpu' and not authorized_real:
        raise PermissionError('GPU work requires explicit authorization')
    bc=Path(config['bc_checkpoint'])
    bc_stats=bc.parent/'vecnormalize.pkl'
    if not bc.exists() or not bc_stats.exists():
        raise FileNotFoundError('Provide BC checkpoint and its matching vecnormalize.pkl')
    bc_metrics=json.loads((bc.parent/'bc_metrics.json').read_text())
    if config.get('mode') == 'real' and bc_metrics['synthetic']:
        raise ValueError('Synthetic BC cannot be presented as real experiment initialization')
    factory=env_factory(config,authorized_real,output/'packets' if config.get('mode')=='real' else None)
    stats=Path(resume)/'vecnormalize.pkl' if resume else bc_stats
    vec=make_vec(factory,stats,training=False)  # fixed BC normalization across all lambdas
    identity=dict(config=signature(config),bc_sha256=file_hash(bc),bc_stats_sha256=file_hash(bc_stats),
                  features=list(FEATURES),gain_table=[list(x) for x in GAIN_TABLE])
    if resume:
        resume=Path(resume)
        previous=json.loads((resume/'manifest.json').read_text())
        for key in ('config','bc_sha256','bc_stats_sha256','features','gain_table'):
            if previous[key] != identity[key]:
                raise ValueError(f'Resume mismatch: {key}; cannot mix lambdas/configurations')
        model=PPO.load(str(resume/'model.zip'),env=vec,device=device)
        model.tensorboard_log=str(output/'tensorboard') if config.get('tensorboard',True) else None
        # Local trusted checkpoint only: Python/NumPy RNG tuples require pickle.
        rng=torch.load(resume/'rng.pt',map_location='cpu',weights_only=False)
        random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'])
        if device != 'cpu' and rng.get('cuda') is not None:
            torch.cuda.set_rng_state_all(rng['cuda'])
        initial_actor=previous['initial_actor_sha256']
    else:
        model=build_ppo(vec,seed=config['seed'],device=device,ppo=config.get('ppo'),
                        tensorboard_log=str(output/'tensorboard') if config.get('tensorboard',True) else None)
        initial_actor=initialize_from_bc(model,bc)
    identity.update(initial_actor_sha256=initial_actor,synthetic=config.get('mode','mock')=='mock',
        seed=config['seed'],resume_from=str(resume) if resume else None,
        resume_limitations='Optimizer/timesteps/RNG/normalizer restored. Simulator, remote VLA, partial episode, rollout and task RNG not serialized; restart at episode boundary, not bit-exact continuation.',
        normalization='frozen BC training-set statistics',critic='fresh at initial RL construction')
    (output/'config.json').write_text(json.dumps(config,indent=2))
    (output/'manifest.json').write_text(json.dumps(identity,indent=2))
    start=time.perf_counter()
    try:
        model.learn(total_timesteps=config['total_timesteps'],reset_num_timesteps=not bool(resume),callback=MetricsCallback(output))
        model.save(output/'model.zip')
        vec.save(output/'vecnormalize.pkl')
        torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all() if device != 'cpu' else None),output/'rng.pt')
        identity.update(timesteps=model.num_timesteps,wall_seconds=time.perf_counter()-start,
                        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/'manifest.json').write_text(json.dumps(identity,indent=2))
    finally:
        vec.close()
    return identity


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ppo.yaml')
    p.add_argument('--resume',help='Prior run directory; output must be a NEW directory')
    p.add_argument('--authorize-real',action='store_true')
    a=p.parse_args()
    train(read_config(a.config),a.authorize_real,a.resume)


if __name__=='__main__':
    main()
