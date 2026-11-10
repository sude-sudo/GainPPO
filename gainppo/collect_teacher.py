"""Collect validated teacher records; synthetic data always carries synthetic=true."""
import argparse
import json
from pathlib import Path
import yaml
from .teacher import GainTeacherPolicy,collect
from .config import read_config, env_factory


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/ppo.yaml');p.add_argument('--rules',default='configs/teacher.yaml')
    p.add_argument('--rd',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3,4,5])
    p.add_argument('--authorize-real',action='store_true')
    a=p.parse_args()
    cfg=read_config(a.config)
    rules=yaml.safe_load(Path(a.rules).read_text())
    teacher=GainTeacherPolicy(rules,json.loads(Path(a.rd).read_text()))
    env=env_factory(cfg,a.authorize_real)()
    try:
        collect(env,teacher,a.output,a.seeds,cfg.get('ecomp_env','mock'),cfg.get('mode','mock')=='mock')
    finally:
        env.close()


if __name__=='__main__':
    main()
