"""Strict real ECD1 packet adapter. Never substitutes likelihood estimates."""
from pathlib import Path
import struct
import time
import numpy as np
import torch
from .observation import rgb_float

HEADER = struct.Struct('>4sQIII')


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def psnr(a, b):
    mse = float(np.mean((rgb_float(a)-rgb_float(b))**2))
    return float(-10*np.log10(max(mse, 1e-12)))  # finite cap 120 dB


class CodecAdapter:
    def __init__(self, model, device='cpu', packet_dir=None):
        self.model = freeze(model.to(device))
        self.device = device
        self.packet_dir = None if packet_dir is None else Path(packet_dir)
        if self.packet_dir:
            self.packet_dir.mkdir(parents=True, exist_ok=True)
        self.reset(0)

    def reset(self, episode_id):
        self.episode_id, self.frame_index, self.state = episode_id, 0, None

    def tensor(self, image):
        a = rgb_float(image)
        if not np.isfinite(a).all():
            raise ValueError('Nonfinite RGB cannot be coded')
        return torch.as_tensor(a, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)

    def process(self, main, wrist, main_rate, wrist_rate):
        t = time.perf_counter()
        with torch.inference_mode():
            packet = self.model.compress(self.tensor(main), self.tensor(wrist),
                episode_id=self.episode_id, frame_index=self.frame_index,
                main_rate=main_rate, wrist_rate=wrist_rate)
            encoded = time.perf_counter()
            if not isinstance(packet, bytes) or len(packet) < HEADER.size:
                raise ValueError('Codec must return complete ECD1 bytes')
            magic, episode, frame, nm, nw = HEADER.unpack_from(packet)
            if magic != b'ECD1' or episode != self.episode_id or frame != self.frame_index or min(nm, nw) <= 0 or HEADER.size+nm+nw != len(packet):
                raise ValueError('Malformed or out-of-sequence ECD1 packet')
            decoded = self.model.decompress(packet, state=self.state)
            self.state = decoded['state']
            rebuilt = [decoded[k].detach().cpu()[0].permute(1, 2, 0).numpy().clip(0, 1)
                       for k in ('main_hat', 'wrist_hat')]
        pixels = [int(np.prod(im.shape[:2])) for im in (main, wrist)]
        if any(a.shape != b.shape for a, b in zip((main, wrist), rebuilt)):
            raise ValueError('Codec reconstruction resolution mismatch')
        if self.packet_dir:
            with (self.packet_dir / f'{episode:06d}_{frame:06d}.ecd').open('xb') as f:
                f.write(packet)
        info = dict(bits=len(packet)*8, pixels=sum(pixels), actual_bpp=8*len(packet)/sum(pixels),
                    view_bpp=dict(main=nm*8/pixels[0], wrist=nw*8/pixels[1]),
                    psnr=dict(main=psnr(main, rebuilt[0]), wrist=psnr(wrist, rebuilt[1])),
                    episode_id=episode, frame_index=frame, bpp_type='actual_packet_bytes',
                    encode_seconds=encoded-t, decode_seconds=time.perf_counter()-encoded)
        self.frame_index += 1
        return *rebuilt, info


def load_codec(checkpoint, model_config, device='cpu'):
    """Source package 'compress' must resolve to audited dual-view Codec tree."""
    from compress.dual_view_codec import DualViewCodec
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = saved.get('model', saved)
    key = 'analysis.gain.log_gain'
    if key not in weights:
        raise ValueError(f'Checkpoint missing {key}; explicit migration outside PPO required')
    levels = weights[key].shape[0]
    if levels < 6 or ('levels' in model_config and model_config['levels'] != levels):
        raise ValueError('Checkpoint levels conflict with configuration or six-level selection')
    config = dict(model_config, levels=levels)
    model = DualViewCodec(**config)
    model.load_state_dict(weights, strict=True)
    model.update(force=True)
    return CodecAdapter(model, device)
