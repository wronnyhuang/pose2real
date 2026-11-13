#!/usr/bin/env python3
"""Make page/dummy/: a fake ROOT (3 clips, 4 runs with blurred 'generated' videos, fake table/spend/md) to test build_page.py --dummy.
Uses ROOT/data/clips/<id>/{lm.npz,audio.wav} when the DATA agent has produced them; otherwise extracts lm.npz with common/lmk.py
and cuts audio from the source video named in render2real/data/pairs/<id>/meta.json."""
import json, os, shutil, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
D = os.path.join(HERE, 'dummy'); PAIRS = __import__('os').path.expandvars('$WORK/data/pairs')
VPY = __import__('os').path.expandvars('$WORK/eval/venv/bin/python')
IDS = ['M000_P0115', 'M009_P0217', 'M023_P0370']
shutil.rmtree(D, ignore_errors=True)
for c in IDS:
    cd = os.path.join(D, 'data', 'clips', c); os.makedirs(cd)
    shutil.copy(os.path.join(PAIRS, c, 'real.mp4'), cd)
    meta = json.load(open(os.path.join(PAIRS, c, 'meta.json')))
    real = os.path.join(ROOT, 'data', 'clips', c)
    if os.path.exists(os.path.join(real, 'audio.wav')): shutil.copy(os.path.join(real, 'audio.wav'), cd)
    else: subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-ss', str(meta['src_start_s']), '-i', meta['src_video'], '-t', '8', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', os.path.join(cd, 'audio.wav')])
    if os.path.exists(os.path.join(real, 'lm.npz')): shutil.copy(os.path.join(real, 'lm.npz'), cd)
    else: subprocess.check_call([VPY, os.path.join(ROOT, 'common', 'lmk.py'), os.path.join(cd, 'real.mp4'), os.path.join(cd, 'lm.npz')])
json.dump({'clips': [{'id': c, 'identity': c.split('_')[1], 'split': 'heldout', 'origin': 'render2real'} for c in IDS]}, open(os.path.join(D, 'data', 'manifest.json'), 'w'))
runs = {  # run -> (group, label, blur sigma, lse_c, lse_d, face, mouth, pose, realism, ms)
    'inj_concat': ('injection', 'Concat injection', 6, 5.1, 8.9, .031, .052, .047, 6.2, 41),
    'inj_cross': ('injection', 'Cross-attention injection', 4, 6.4, 8.1, .027, .044, .041, 6.9, 55),
    'look_0': ('lookahead', 'Look-ahead 0 frames', 8, 4.2, 9.6, .035, .061, .05, 5.8, 41),
    'enc_wavlm': ('encoder', 'WavLM encoder', 3, 7.0, 7.7, .024, .039, .04, 7.3, 62),
}
rows = [{'run': 'real', 'lse_c': 8.4, 'lse_d': 6.9, 'face_nme': 0, 'mouth_nme': 0, 'pose_err': 0, 'realism': 9.1, 'fooled_rate': 0.5, 'det_rate': 1.0, 'offset': 0},
        {'run': 'render2real_fast', 'syncnet': {'lse_c': 5.5, 'lse_d': 8.7}, 'landmarks': {'face_nme': .03, 'mouth_nme': .05, 'pose_err': .045, 'det_rate': .99}, 'realism': {'score_1_10': 6.5, 'fooled_rate': .1}, 'ms_per_frame': 150}]
for r, (g, lab, sig, c, d, fn, mn, pe, re_, ms) in runs.items():
    od = os.path.join(D, 'runs', r, 'out'); os.makedirs(od)
    json.dump({'group': g, 'label': lab, 'backbone': 'SD-Turbo', 'injection': r.split('_')[1] if g == 'injection' else 'cross',
               'encoder': 'wavlm' if g == 'encoder' else 'mel', 'audio_lookahead_frames': 0 if r == 'look_0' else 4, 'lm_lookahead_frames': 2,
               'ms_per_frame': ms, 'train_minutes': 95, 'steps': 12000}, open(os.path.join(D, 'runs', r, 'run.json'), 'w'))
    for c_ in IDS:
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(D, 'data', 'clips', c_, 'real.mp4'), '-vf', f'gblur=sigma={sig}', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(od, c_ + '.mp4')])
    rows.append({'run': r, 'lse_c': c, 'lse_d': d, 'offset': 0.5, 'face_nme': fn, 'mouth_nme': mn, 'pose_err': pe, 'mouth_open_corr': 0.8, 'det_rate': 0.98, 'realism': re_, 'fooled_rate': 0.05})
os.makedirs(os.path.join(D, 'eval', 'results')); json.dump(rows, open(os.path.join(D, 'eval', 'results', 'table.json'), 'w'))
os.makedirs(os.path.join(D, 'pods')); json.dump({'usd': 7.8, 'cap_usd': 30, 'pods': [{'name': 'train-a', 'gpu': 'A100 80GB', 'rate': 1.19, 'hours': 6.5, 'usd': 7.74}]}, open(os.path.join(D, 'pods', 'spend.json'), 'w'))
w = lambda n, t: open(os.path.join(D, n), 'w').write(t)
w('summary.md', 'DUMMY DATA. Pipeline test only.\n\nThe best fake run is **WavLM encoder**. Lip sync gets close to real.\n\n- Fake bullet one\n- Fake bullet **two**\n')
w('worked.md', '- Cross-attention beat concat.\n- **WavLM** audio features.\n'); w('didnt.md', '- Zero look-ahead hurt sync.\n'); w('next.md', '- Train longer.\n')
json.dump({'runs': [{'run': 'enc_wavlm', 'title': 'WavLM encoder', 'note': 'best sync'}, {'run': 'inj_cross', 'title': 'Cross-attention'}, {'run': 'look_0', 'title': 'No look-ahead', 'note': 'worst'}], 'clips': IDS[:3]}, open(os.path.join(D, 'showcase.json'), 'w'))
open(os.path.join(D, 'DUMMY_READY'), 'w').write('ok')
print('dummy ready', D)
