"""Stage 1: balanced two-head CE, actor only; critic has no value learning."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .policy import build_ppo, make_vec
from .mock_env import make_mock_env
from .teacher import load_rows, validate_rows


def split_rows(rows, seed=42, validation_fraction=.2):
    # Group by seed (stronger than episode split); all tasks/episodes of a seed stay together.
    groups = sorted({r['seed'] for r in rows})
    if len(groups) < 2:
        raise ValueError('Need at least two independent seeds for held-out validation')
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    val = set(groups[:max(1, min(len(groups)-1, round(len(groups)*validation_fraction)))])
    return [r for r in rows if r['seed'] not in val], [r for r in rows if r['seed'] in val]


def class_weights(labels):
    counts = np.bincount(labels, minlength=11)
    present = counts > 0
    weights = np.zeros(11, np.float32)
    weights[present] = len(labels)/(present.sum()*counts[present])
    return counts, weights


def actor_parameters(policy):
    return list(policy.mlp_extractor.policy_net.parameters())+list(policy.action_net.parameters())


def actor_logits(policy, obs):
    features = policy.extract_features(obs)
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def bc_step(policy, obs, labels, weights, optimizer):
    logits = actor_logits(policy, obs).split(11, dim=-1)
    loss = sum(F.cross_entropy(branch, labels[:, i], weight=weights[i]) for i, branch in enumerate(logits))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def validation(policy, x, y):
    with torch.no_grad():
        predictions = torch.stack([z.argmax(-1) for z in actor_logits(policy,x).split(11,dim=-1)],dim=1).cpu().numpy()
    confusions, recalls = [], []
    for i in range(2):
        cm = np.zeros((11,11),int)
        np.add.at(cm,(y[:,i],predictions[:,i]),1)
        support = cm.sum(1)
        recall = np.divide(cm.diagonal(),support,out=np.zeros(11,float),where=support>0)
        confusions.append(cm.tolist())
        recalls.append(float(recall[support>0].mean()))
    return dict(confusion_matrices=confusions, macro_recall=recalls)


def train_bc(rows, output, seed=42, epochs=100, patience=10, batch_size=128, learning_rate=1e-4):
    validate_rows(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    train, val = split_rows(rows,seed)
    vec = make_vec(make_mock_env)
    # Fit once on training records only. Freeze this exact normalizer in BC and RL.
    train_x = np.asarray([r['obs'] for r in train],np.float32)
    vec.obs_rms.update(train_x)
    vec.training = False
    model = build_ppo(vec,seed=seed)
    policy = model.policy
    policy.requires_grad_(False)
    params = actor_parameters(policy)
    for p in params:
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(params,lr=learning_rate)
    x = torch.as_tensor(vec.normalize_obs(train_x.copy()),dtype=torch.float32)
    y = torch.as_tensor([r['action'] for r in train],dtype=torch.long)
    vx = torch.as_tensor(vec.normalize_obs(np.asarray([r['obs'] for r in val],np.float32)),dtype=torch.float32)
    vy = np.asarray([r['action'] for r in val],np.int64)
    stats = [class_weights(y[:,i].numpy()) for i in range(2)]
    weights = [torch.as_tensor(s[1]) for s in stats]
    best, stale, history = -1., 0, []
    for epoch in range(epochs):
        order = torch.randperm(len(train))
        losses = []
        for index in order.split(batch_size):
            losses.append(bc_step(policy,x[index],y[index],weights,optimizer))
        metrics = validation(policy,vx,vy)
        score = np.mean(metrics['macro_recall'])
        history.append(dict(epoch=epoch, loss=float(np.mean(losses)), **metrics))
        if score > best:
            best, stale = score, 0
            best_state = copy.deepcopy(policy.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    policy.load_state_dict(best_state)
    # Save ordinary PPO container; fresh RL models copy only actor tensors.
    model.save(output/'bc_actor.zip')
    vec.save(output/'vecnormalize.pkl')
    metrics = dict(synthetic=bool(rows[0]['synthetic']), critic_value_trained=False,
        feature_order=list(__import__('gainppo.observation',fromlist=['FEATURES']).FEATURES),
        train_seeds=sorted({r['seed'] for r in train}), validation_seeds=sorted({r['seed'] for r in val}),
        counts=[s[0].tolist() for s in stats], weights=[s[1].tolist() for s in stats],
        missing_classes=[np.flatnonzero(s[0]==0).tolist() for s in stats],
        history=history, best_validation=validation(policy,vx,vy),
        normalization='training records only; frozen for all subsequent RL/evaluation')
    (output/'bc_metrics.json').write_text(json.dumps(metrics,indent=2))
    for i,view in enumerate(('main','wrist')):
        np.savetxt(output/f'{view}_confusion.csv',metrics['best_validation']['confusion_matrices'][i],fmt='%d',delimiter=',')
    vec.close()
    return metrics


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config', default='configs/bc.yaml')
    args=p.parse_args()
    import yaml
    cfg=yaml.safe_load(Path(args.config).read_text())
    train_bc(load_rows(cfg.pop('dataset')),**cfg)


if __name__ == '__main__':
    main()
