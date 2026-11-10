"""Matched seeds, common BC artifact, lambda is the only experiment difference."""
import argparse
import copy
import json
from pathlib import Path
from .config import read_config
from .train_ppo import train


def sweep_configs(config):
    root=Path(config['output'])
    seen=set()
    for seed in config.get('seeds',[config['seed']]):
        for value in config['lambda_bpp_values']:
            if value < 0:
                raise ValueError('lambda must be nonnegative')
            name=f'lambda_{value:g}_seed_{seed}'
            if name in seen:
                raise ValueError('Duplicate seed/lambda run')
            seen.add(name)
            c=copy.deepcopy(config)
            for key in ('lambda_bpp_values','seeds'):
                c.pop(key,None)
            c.update(lambda_bpp=float(value),seed=seed,output=str(root/name))
            yield name,c


def train_lambda_sweep(config,authorized_real=False):
    root=Path(config['output'])
    configs=list(sweep_configs(config))
    root.mkdir(parents=True,exist_ok=False)
    records={}
    for name,c in configs:
        records[name]=train(c,authorized_real)
        (root/'sweep_manifest.json').write_text(json.dumps(records,indent=2))
    return records


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/lambda_sweep.yaml')
    p.add_argument('--authorize-real',action='store_true')
    a=p.parse_args()
    train_lambda_sweep(read_config(a.config),a.authorize_real)


if __name__=='__main__':
    main()
