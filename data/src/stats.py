"""Numbers for REPORT.md -> work/stats.json + printed summary."""
import json, glob, collections, numpy as np
D = __import__('os').path.expandvars('$WORK/data')
man = json.load(open(f'{D}/manifest.json'))
out = {'counts': collections.Counter(c['split'] for c in man['clips'])}
per = collections.defaultdict(collections.Counter)
for c in man['clips']: per[c['identity']][c['split']] += 1
out['per_identity'] = {k: dict(v) for k, v in sorted(per.items())}
det = collections.defaultdict(list)
for c in man['clips']:
    d = np.load(f"{D}/clips/{c['id']}/lm.npz")
    k = c['origin'] if c['split'] == 'train' else c['split']
    det[k].append((d['face_ok'].mean(), d['pose_ok'].mean(), d['pose'][d['pose_ok']][:, [11, 12], 3].mean() if d['pose_ok'].any() else 0,
                   d['face_ok'].shape[0]))
out['detection'] = {k: dict(n=len(v), face_ok_mean=float(np.mean([x[0] for x in v])), face_ok_min=float(np.min([x[0] for x in v])),
                            pose_ok_mean=float(np.mean([x[1] for x in v])), pose_ok_min=float(np.min([x[1] for x in v])),
                            shoulder_vis_mean=float(np.mean([x[2] for x in v])), frames_all_240=bool(all(x[3] == 240 for x in v))) for k, v in det.items()}
out['dropped'] = man['dropped']
json.dump(out, open(f'{D}/work/stats.json', 'w'), indent=1, default=int)
print(json.dumps({k: out[k] for k in ('counts', 'detection')}, indent=1, default=int)); print('dropped', len(out['dropped']))
for x in out['dropped']: print(' ', x)
