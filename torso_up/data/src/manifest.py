"""(Re)build manifest.json from clips/ + work/plan.json (atomic). Heldout clips are always kept (reported if QC flags them);
other clips need qc.json with an empty drop. usage: manifest.py [--final]"""
import json, os, sys, collections, numpy as np
D = __import__('os').path.expandvars('$WORK/data')
HELDOUT = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
FILES = ('real.mp4', 'audio.wav', 'audio_pad.wav', 'lm.npz', 'meta.json')
plan = json.load(open(f'{D}/work/plan.json'))
clips, dropped, pending = [], [], []
for w in plan:
    cd = f"{D}/clips/{w['id']}"
    if not os.path.exists(f'{cd}/qc.json') or not all(os.path.exists(f'{cd}/{f}') for f in FILES): pending.append(w['id']); continue
    m = json.load(open(f'{cd}/meta.json'))
    q = m['qc']   # drop rule re-derived from the QC numbers (second-face threshold 0.3, see proc.py)
    q['drop'] = '; '.join(r for c, r in ((q['frames'] != 240, f"frames {q['frames']}"), (q['face_ok'] < 0.95, f"face_ok {q['face_ok']:.2f}"),
        (q['face_inside'] < 0.95, f"face leaves crop {q['face_inside']:.2f}"), (q['second_face'] > 0.3, f"second face {q['second_face']:.2f}"),
        (q['pose_ok'] < 0.9, f"pose_ok {q['pose_ok']:.2f}"), (q['shoulders_inside'] < 0.9, f"shoulders leave crop {q['shoulders_inside']:.2f}")) if c)
    if m['qc']['drop'] and m['split'] != 'heldout':
        dropped.append(dict(id=w['id'], split=m['split'], set=m['set'], why=m['qc']['drop'])); continue
    if m['split'] == 'train': assert m['identity'] not in HELDOUT, w['id']
    clips.append(dict(id=m['id'], identity=m['identity'], split=m['split'], set=m['set'], origin=m['origin'], hand_frac=m['hand_frac'],
                      hand2_frac=m['hand2_frac'], hand_motion=m['hand_motion'], face_width_px=m['face_width_px'], scale=m['crop']['scale']))
counts = dict(collections.Counter(c['split'] for c in clips))
counts.update({f"{s}/{t}": n for (s, t), n in collections.Counter((c['split'], c['set']) for c in clips).items()})
def st(cs):
    if not cs: return {}
    h = np.array([c['hand_frac'] for c in cs]); h2 = np.array([c['hand2_frac'] for c in cs])
    return dict(n=len(cs), hand_frac_mean=round(float(h.mean()), 3), hand_frac_median=round(float(np.median(h)), 3),
                hand_frac_p10=round(float(np.percentile(h, 10)), 3), share_clips_hand_ge_0p9=round(float((h >= 0.9).mean()), 3),
                hand2_frac_mean=round(float(h2.mean()), 3), hand2_frac_median=round(float(np.median(h2)), 3),
                hand_motion_median=round(float(np.median([c['hand_motion'] for c in cs])), 3),
                face_px_median=round(float(np.median([c['face_width_px'] for c in cs])), 1))
stats = {k: st([c for c in clips if f(c)]) for k, f in (
    ('heldout/same12', lambda c: c['set'] == 'same12'), ('heldout/gesture8', lambda c: c['set'] == 'gesture8'),
    ('heldout_ref', lambda c: c['split'] == 'heldout_ref'), ('train', lambda c: c['split'] == 'train'),
    ('train/prior', lambda c: c['split'] == 'train' and c['set'] == 'prior'), ('train/gesture', lambda c: c['split'] == 'train' and c['set'] == 'gesture'))}
man = dict(clips=clips, status='final' if '--final' in sys.argv else 'growing', heldout_identities=HELDOUT, counts=counts,
           hand_stats=stats, dropped=dropped, pending=len(pending),
           format='clips/<id>/{real.mp4 512x640 30fps 240f, audio.wav 16k 8.000s, audio_pad.wav 16k 10.000s (1 s before/after), lm.npz (common/lmk.py), meta.json}')
json.dump(man, open(f'{D}/manifest.json.tmp', 'w'), indent=1); os.replace(f'{D}/manifest.json.tmp', f'{D}/manifest.json')
print(counts, 'dropped', len(dropped), 'pending', len(pending))
for k, v in stats.items(): print(k, v)
