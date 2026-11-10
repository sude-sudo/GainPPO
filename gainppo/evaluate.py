"""Deterministic matched-seed closed-loop evaluation, BC/fixed-gain baselines."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from .config import read_config,env_factory
from .policy import make_vec


def evaluate(config, run, output, seeds, authorized_real=False, fixed_action=None, bc=False):
    if not seeds:
        raise ValueError("Evaluation needs at least one seed")
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    vec=make_vec(env_factory(config,authorized_real,out/'packets' if config.get('mode')=='real' else None),
                 Path(run)/'vecnormalize.pkl',training=False)
    model=None if fixed_action is not None else PPO.load(str(Path(run)/('bc_actor.zip' if bc else 'model.zip')),device='cpu')
    episodes=[]
    raw_env=vec.venv.envs[0]
    try:
        for seed in seeds:
            obs,_=raw_env.reset(seed=seed)
            while True:
                action=np.asarray(fixed_action,dtype=np.int64) if fixed_action is not None else model.predict(vec.normalize_obs(obs.copy()),deterministic=True)[0]
                obs,reward,terminated,truncated,info=raw_env.step(action)
                if terminated or truncated:
                    episodes.append(dict(seed=seed,**info['episode_metrics']))
                    break
    finally:
        vec.close()
    total_bits=sum(x['bits'] for x in episodes)
    total_pixels=sum(x['pixels'] for x in episodes)
    summary=dict(episodes=episodes,success_rate=float(np.mean([x['success'] for x in episodes])),
                 avg_bpp=total_bits/total_pixels,mean_episode_bpp=float(np.mean([x['avg_bpp'] for x in episodes])),
                 synthetic=config.get('mode','mock')=='mock', fixed_action=fixed_action,
                 gain_histogram=np.sum([x['gain_histogram'] for x in episodes],axis=0).tolist(),
                 view_bpp={v:float(np.mean([x['view_bpp'][v] for x in episodes])) for v in ('main','wrist')},
                 psnr={v:float(np.mean([x['psnr'][v] for x in episodes])) for v in ('main','wrist')},
                 note='avg_bpp is pooled bits/pixels; terminal objective uses mean_episode_bpp')
    (out/'evaluation.json').write_text(json.dumps(summary,indent=2))
    return summary


def plot_curves(evaluations, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots()
    for label,path in evaluations.items():
        r=json.loads(Path(path).read_text())
        ax.scatter(r['avg_bpp'],r['success_rate'],label=label)
    ax.set(xlabel='Actual BPP (synthetic when marked mock)',ylabel='Success rate',ylim=(0,1.05))
    ax.legend(); fig.tight_layout(); fig.savefig(output); plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--run',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seeds',type=int,nargs='+',default=[101,102,103])
    p.add_argument('--fixed-action',type=int,nargs=2);p.add_argument('--bc',action='store_true')
    p.add_argument('--authorize-real',action='store_true')
    a=p.parse_args()
    evaluate(read_config(a.config),a.run,a.output,a.seeds,a.authorize_real,a.fixed_action,a.bc)


if __name__=='__main__':
    main()
