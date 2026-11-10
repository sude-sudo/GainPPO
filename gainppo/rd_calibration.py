"""Calibrate actual bytes and per-view PSNR for all 11 selectors on fixed RGB pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .actions import GAIN_TABLE


def calibrate(codec, pairs, output, checkpoint_sha256=None):
    pairs=list(pairs)
    if not pairs:
        raise ValueError('Calibration needs fixed validation image pairs')
    result=dict(main=[],wrist=[],paired=[],bpp_type='actual_packet_bytes',checkpoint_sha256=checkpoint_sha256)
    for action,rate in enumerate(GAIN_TABLE):
        codec.reset(action)
        infos=[]
        for main,wrist in pairs:
            _,_,info=codec.process(main,wrist,rate,rate)
            if info['bpp_type'] != 'actual_packet_bytes':
                raise ValueError('Calibration requires real packets')
            infos.append(info)
        bpp=sum(x['bits'] for x in infos)/sum(x['pixels'] for x in infos)
        result['paired'].append(dict(action=action,rate=list(rate),bpp=bpp))
        for view in ('main','wrist'):
            result[view].append(dict(action=action,bpp=float(np.mean([x['view_bpp'][view] for x in infos])),
                                     psnr=float(np.mean([x['psnr'][view] for x in infos]))))
    result['B_ref']=float(np.median([x['bpp'] for x in result['paired']]))
    result['definition']='B_ref median paired actual BPP over 11 rates; view BPP excludes shared ECD1 header; PSNR capped at 120dB'
    with Path(output).open('x') as f:
        json.dump(result,f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--images',required=True,help='NPZ main/wrist uint8 N,H,W,3')
    p.add_argument('--model-config',required=True,help='JSON with exact decoder_config/history_size/options')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--authorize-real',action='store_true')
    a=p.parse_args()
    if not a.authorize_real:
        p.error('Real calibration requires explicit --authorize-real')
    from .codec_adapter import load_codec
    codec=load_codec(a.checkpoint,json.loads(Path(a.model_config).read_text()),a.device)
    data=np.load(a.images,allow_pickle=False)
    if len(data['main']) != len(data['wrist']):
        raise ValueError('Paired image count mismatch')
    digest=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    calibrate(codec,zip(data['main'],data['wrist']),a.output,digest)


if __name__=='__main__':
    main()
