"""Small CPU BC -> two independent PPOs -> matched-seed evaluation. No real claims."""
import argparse
import json
from pathlib import Path
import torch
from .mock_env import make_mock_env
from .teacher import GainTeacherPolicy, collect, load_rows
from .train_bc import train_bc
from .train_sweep import train_lambda_sweep
from .evaluate import evaluate, plot_curves


def run(output):
    torch.set_num_threads(1)
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    rd={v:[dict(action=i,bpp=float((i+3)%11+1),psnr=float(20+i*2)) for i in range(11)] for v in ('main','wrist')}
    rd['bpp_type']='synthetic_mock'
    rules=dict(version='synthetic-smoke-v1',allow_synthetic=True,motion_weights=[5,5,1],
               demand_thresholds=[.15,.5],minimum_psnr=[25,30,35])
    env=make_mock_env()
    try:
        collect(env,GainTeacherPolicy(rules,rd),root/'teacher_synthetic.jsonl',range(8),'mock',True)
    finally:
        env.close()
    train_bc(load_rows(root/'teacher_synthetic.jsonl'),root/'bc',epochs=3,patience=2,batch_size=16)
    cfg=dict(mode='mock',output=str(root/'sweep'),seed=42,seeds=[42],device='cpu',tensorboard=False,
             bc_checkpoint=str(root/'bc/bc_actor.zip'),lambda_bpp_values=[0.,.5],lambda_bpp=.1,
             B_ref=1.,w_s=1.,shared_gain=False,reward_mode='terminal',total_timesteps=32,
             ppo=dict(n_steps=16,batch_size=16,n_epochs=2))
    records=train_lambda_sweep(cfg)
    curves={}
    for name in records:
        evaluate(cfg,root/'sweep'/name,root/f'eval_{name}',[101,102])
        curves[name]=str(root/f'eval_{name}'/'evaluation.json')
    plot_curves(curves,root/'synthetic_curve.png')
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='outputs/smoke')
    a=p.parse_args();run(a.output)
    print(f'SYNTHETIC CPU smoke complete: {a.output}')


if __name__=='__main__':
    main()
