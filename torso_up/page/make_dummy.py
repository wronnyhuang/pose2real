#!/usr/bin/env python3
"""Make page/dummy/: a fake ROOT for `build_page.py --dummy`. Uses PRIOR (pose2real) bust clips letterboxed to 512x640 as stand-ins,
fake hands drawn around the wrists, blurred 'generated' videos, a fake table.json in the contract's shape, bust_* rows copied from PRIOR."""
import json, os, shutil, subprocess, math, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, 'dummy')
PRIOR = __import__('os').path.expandvars('$WORK')
CLIPS = {'M000_P0115': 'M000_P0115', 'M075_P0845': 'M075_P0845', 'G00_P0370': 'M023_P0370', 'G01_P1174': 'M103_P1174'}  # new id -> prior clip
SETS = {'M000_P0115': 'same12', 'M075_P0845': 'same12', 'G00_P0370': 'gesture8', 'G01_P1174': 'gesture8'}
shutil.rmtree(D, ignore_errors=True)


def fake_hand(wrist, t, side):
    pts = np.zeros((21, 3), np.float32)
    pts[0, :2] = wrist
    for f in range(5):
        ang = math.radians(-90 + (f - 2) * 20 + (-25 if f == 0 else 0) * (1 if side else -1) + 12 * math.sin(t / 7 + f + side))
        for k in range(4):
            r = 0.012 + (0.016 if f else 0.011) * (k + 1) * (1 - 0.18 * math.sin(t / 9 + f))
            idx = 1 + f * 4 + k
            pts[idx, 0] = wrist[0] + r * math.cos(ang) * (640 / 512)
            pts[idx, 1] = wrist[1] + r * math.sin(ang)
    return pts


for new, old in CLIPS.items():
    cd = os.path.join(D, 'data', 'clips', new); os.makedirs(cd)
    src = os.path.join(PRIOR, 'data', 'clips', old)
    subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(src, 'real.mp4'), '-vf', 'scale=512:512,pad=512:640:0:64:black,fps=30',
                           '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', os.path.join(cd, 'real.mp4')])
    shutil.copy(os.path.join(src, 'audio.wav'), cd)
    d = dict(np.load(os.path.join(src, 'lm.npz')))
    for k in ('face', 'pose'): d[k][..., 1] = (d[k][..., 1] * 512 + 64) / 640
    T = len(d['face']); hands = np.zeros((T, 2, 21, 3), np.float32); hok = np.zeros((T, 2), bool)
    for t in range(T):
        for s in (0, 1):
            if (t // 40 + s) % 3 == 2: continue      # hidden stretches, to test "hidden when not tracked"
            w = (0.3 + 0.4 * s + 0.06 * math.sin(t / 15 + s), 0.78 + 0.05 * math.sin(t / 11 + 2 * s))
            hands[t, s] = fake_hand(w, t, s); hok[t, s] = True
    d.update(hands=hands, hands_ok=hok, hands_score=hok.astype(np.float32))
    np.savez_compressed(os.path.join(cd, 'lm.npz'), **d)
    json.dump({'id': new, 'set': SETS[new]}, open(os.path.join(cd, 'meta.json'), 'w'))
json.dump({'clips': [{'id': c, 'identity': c.split('_')[1], 'split': 'heldout', 'set': SETS[c], 'hand_frac': 0.55 + 0.1 * i}
                     for i, c in enumerate(CLIPS)]}, open(os.path.join(D, 'data', 'manifest.json'), 'w'))
runs = {'fake_base': ('baseline', 'Torso baseline', 6, 5.0, 9.0, 1.2), 'fake_hands': ('hands in conditioning', 'Hands fed to encoder', 3, 6.1, 8.2, 1.0),
        'fake_big': ('resolution', 'Bigger backbone', 2, 6.5, 8.0, 1.0)}
rows = []
for name, r in (('real', 0), ('real_crf28', 1)):
    rows.append({'run': name, 'mean': {'lse_c': 7.9 - r * .3, 'lse_d': 6.9, 'face_nme': 0.4 * r, 'mouth_nme': .5 * r, 'pose_err': .3 * r, 'arm_err': .5 * r, 'hand_err': .6 * r,
                 'hand_det_rate': .93, 'hand_extra_rate': .02, 'finger_pose_err': 0.5 * r, 'realism': 8.4, 'hands_score_1_10': 8.0, 'fooled_rate': .5},
                 'mean_same12': {'lse_c': 7.8}})
for r, (g, lab, sig, c, dd, ms) in runs.items():
    od = os.path.join(D, 'runs', r, 'out'); os.makedirs(od)
    json.dump({'group': g, 'label': lab, 'ms_per_frame': 40 * ms, 'gpu': 'A100 80GB', 'train_minutes': 120}, open(os.path.join(D, 'runs', r, 'run.json'), 'w'))
    for c_ in CLIPS:
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-i', os.path.join(D, 'data', 'clips', c_, 'real.mp4'), '-vf', f'gblur=sigma={sig}', '-an',
                               '-c:v', 'libx264', '-pix_fmt', 'yuv420p', os.path.join(od, c_ + '.mp4')])
    m = {'lse_c': c, 'lse_d': dd, 'face_nme': 3.1 - sig / 10, 'mouth_nme': 4.2, 'pose_err': 4.0, 'shoulder_err': 2.2, 'elbow_err': 5.1, 'wrist_err': 7.2, 'arm_err': 6.1,
         'hand_err': 11.0 - sig / 2, 'hand_err_hs': 45 - sig, 'hand_det_rate': .6 + sig / 40, 'hand_extra_rate': .05, 'finger_pose_err': 30 - sig, 'finger_angle_err_deg': 24 - sig,
         'realism': 5.5 + sig / 3, 'hands_score_1_10': 3.5, 'fooled_rate': .05, 'offset': .5}
    if r == 'fake_big': m.pop('hands_score_1_10'); m.pop('finger_angle_err_deg')
    rows.append({'run': r, 'mean': m, 'mean_same12': dict(m, lse_c=c + .1), 'mean_gesture8': m})
prior = json.load(open(os.path.join(PRIOR, 'eval', 'results', 'table.json')))['rows']
for r in prior:
    if r['run'] in ('v3_seen', 'v4_more', 'base'):
        rows.append({**{k: v for k, v in r.items() if k != 'run'}, 'run': 'bust_' + r['run']})
os.makedirs(os.path.join(D, 'eval', 'results')); json.dump({'rows': rows}, open(os.path.join(D, 'eval', 'results', 'table.json'), 'w'))
os.makedirs(os.path.join(D, 'pods')); json.dump({'usd': 7.8, 'cap_usd': 30, 'pods': [{'name': 'train-a', 'gpu': 'A100 80GB', 'rate': 1.19, 'hours': 6.5, 'usd': 7.74, 'terminated': True, 'note': 'fake'}]}, open(os.path.join(D, 'pods', 'spend.json'), 'w'))
w = lambda n, t: open(os.path.join(D, n), 'w').write(t)
w('summary.md', 'DUMMY DATA. Pipeline test only.\n\nThe best fake run is **Hands fed to encoder**. Lip sync gets close to real.\n\n- Fake bullet one\n- Fake bullet **two**\n')
w('worked.md', '- Feeding hands helped.\n- **Bigger** backbone slightly.\n\n| model | ms |\n|---|---|\n| a | 40 |\n| b | 55 |\n')
w('didnt.md', '- Nothing about fingers.\n'); w('speed.md', 'Lower resolution is **faster**.\n'); w('next.md', '- Train longer.\n')
json.dump([{'run': 'fake_hands', 'label': 'Hands fed to encoder', 'clips': list(CLIPS)}, {'run': 'fake_base', 'label': 'Torso baseline', 'clips': list(CLIPS)[:2]}],
          open(os.path.join(D, 'showcase.json'), 'w'))
open(os.path.join(D, 'DUMMY_READY'), 'w').write('ok')
print('dummy ready', D)
