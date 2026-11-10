"""Reproducible heuristic teacher; calibrated RD labels are not optimal truths."""
import json
from pathlib import Path
import numpy as np
from .actions import parse_action
from .observation import observation_space


class GainTeacherPolicy:
    def __init__(self, rules, rd):
        self.rules = rules
        self.version = rules['version']
        if rd.get('bpp_type') != 'actual_packet_bytes' and not rules.get('allow_synthetic', False):
            raise ValueError('Real teacher requires real RD calibration')
        self.rd = rd
        for view in ('main', 'wrist'):
            if len(rd[view]) != 11 or sorted(x['action'] for x in rd[view]) != list(range(11)):
                raise ValueError('RD table must cover all 11 actions for each view')

    def predict(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if not observation_space().contains(obs) or not np.isfinite(obs).all():
            raise ValueError('Teacher requires six valid ordered features')
        motion = float(np.dot(obs[2:5], self.rules['motion_weights']))
        actions = []
        for i, view in enumerate(('main', 'wrist')):
            demand = (1-float(obs[i]))*.5 + motion
            tier = int(np.searchsorted(self.rules['demand_thresholds'], demand, side='right'))
            quality = self.rules['minimum_psnr'][tier]
            candidates = [r for r in self.rd[view] if r['psnr'] >= quality]
            row = min(candidates, key=lambda r:(r['bpp'], r['action'])) if candidates else max(self.rd[view], key=lambda r:(r['psnr'], -r['bpp']))
            actions.append(row['action'])
        return np.asarray(actions, np.int64)


def validate_rows(rows):
    if not rows:
        raise ValueError('Empty teacher dataset')
    required = {'episode_id','step','obs','action','rule_version','validity','task_id','seed','synthetic'}
    seen = set()
    for row in rows:
        if not required <= row.keys():
            raise ValueError(f'Missing schema fields: {required-row.keys()}')
        x = np.asarray(row['obs'], np.float32)
        if not observation_space().contains(x) or not np.isfinite(x).all():
            raise ValueError('Invalid teacher observation')
        parse_action(row['action'])
        key = (row['task_id'], row['seed'], row['episode_id'], row['step'])
        if key in seen:
            raise ValueError('Duplicate trajectory step')
        seen.add(key)
    if len({bool(r['synthetic']) for r in rows}) != 1:
        raise ValueError('Cannot mix synthetic and real records')
    return rows


def load_rows(path):
    return validate_rows([json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()])


def collect(env, teacher, path, seeds, task_id, synthetic=False):
    """Use real env only after external authorization and calibration."""
    with Path(path).open('x') as f:
        for episode, seed in enumerate(seeds):
            obs, info = env.reset(seed=seed)
            step = 0
            while True:
                action = teacher.predict(obs)
                row = dict(episode_id=episode, step=step, obs=obs.tolist(), action=action.tolist(),
                    rule_version=teacher.version, validity=info['feature_validity'], task_id=task_id,
                    seed=seed, synthetic=synthetic)
                f.write(json.dumps(row)+'\n')
                obs, _, term, trunc, info = env.step(action)
                step += 1
                if term or trunc:
                    break
