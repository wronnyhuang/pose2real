"""Compute torso crops for all PRIOR windows -> work/prior_crops.json"""
import json, sys, numpy as np
sys.path.insert(0, '.')
import framing, cut
P = __import__('os').path.expandvars('$WORK/data')
out = {}
for c in json.load(open(P + '/manifest.json'))['clips']:
    m = json.load(open(f"{P}/clips/{c['id']}/meta.json"))
    kp, vb = cut.kp_window(m['src_video'], m['src_start_frame'])
    out[c['id']] = dict(split=c['split'], identity=c['identity'], origin=c['origin'], src_video=m['src_video'], st=m['src_start_frame'], crop=framing.crop(kp, vb))
json.dump(out, open('../work/prior_crops.json', 'w'), indent=0)
cr = [v['crop'] for v in out.values() if v['crop']]
print('n', len(out), 'none', sum(v['crop'] is None for v in out.values()))
for k in ('scale', 'w', 'Hf', 'SW'): print(k, np.percentile([c[k] for c in cr], [0, 5, 25, 50, 75, 95, 100]).round(2))
print('head_cut', sum(c['head_cut'] for c in cr), 'scale at 0.5', sum(c['scale'] <= 0.5 for c in cr), 'at 1.4', sum(c['scale'] >= 1.39 for c in cr))
print('bottom clipped', sum(c['y'] + c['h'] < c['bot_raw'] - 1 for c in cr))
