"""Factories deliberately delay GPU/simulator/network access until authorization."""
from pathlib import Path
import importlib
import json
import yaml
from .mock_env import make_mock_env


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def env_factory(config, authorized_real=False, packet_dir=None):
    reward = {k:config[k] for k in ('lambda_bpp','B_ref','w_s','shared_gain','reward_mode') if k in config}
    if config.get('mode','mock') == 'mock':
        return lambda: make_mock_env(**reward)
    if not authorized_real:
        raise PermissionError('Real Codec/VLA/EComp run requires --authorize-real after device approval')
    if config.get('mode') != 'real':
        raise ValueError('mode must be mock or real')
    if config.get('decision_frequency', 'vla_request') != 'vla_request':
        raise ValueError('Only audited VLA-request scheduling frequency is supported')
    if not config.get('calibration'):
        raise ValueError('Real training requires RD calibration artifact')
    calibration=json.loads(Path(config['calibration']).read_text())
    if calibration['bpp_type'] != 'actual_packet_bytes' or calibration['B_ref'] != config['B_ref']:
        raise ValueError('Calibration must contain actual BPP and matching fixed B_ref')
    import hashlib
    digest=hashlib.sha256(Path(config['codec_checkpoint']).read_bytes()).hexdigest()
    if calibration.get('checkpoint_sha256') != digest:
        raise ValueError('RD calibration belongs to a different Codec checkpoint')
    if not config.get('vla_checkpoint'):
        raise ValueError('Record frozen VLA checkpoint provenance')
    target=config['backend_factory']
    module,name=target.split(':')
    def factory():
        from .env import GainEnv
        from .codec_adapter import load_codec
        backend=getattr(importlib.import_module(module),name)(config)
        codec=load_codec(config['codec_checkpoint'],config['codec_model'],config['device'])
        codec.packet_dir=Path(packet_dir) if packet_dir else None
        if codec.packet_dir:
            codec.packet_dir.mkdir(parents=True,exist_ok=False)
        return GainEnv(backend,codec,**reward)
    return factory
