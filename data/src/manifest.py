"""(Re)build manifest.json from clips/ (atomic). Seed clips always; expand/ref clips once qc.json passes. usage: manifest.py [--final]"""
import json, os, glob, sys, collections
D = __import__('os').path.expandvars('$WORK/data')
HELDOUT = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
FILES = ('real.mp4', 'audio.wav', 'audio_pad.wav', 'lm.npz', 'meta.json')
clips, dropped = [], []
for cd in sorted(glob.glob(f'{D}/clips/*')):
    cid = os.path.basename(cd)
    if not all(os.path.exists(f'{cd}/{f}') for f in FILES): continue
    m = json.load(open(f'{cd}/meta.json'))
    if m['origin'] != 'render2real':
        if not os.path.exists(f'{cd}/qc.json'): continue
        q = json.load(open(f'{cd}/qc.json'))
        if q['drop']: dropped.append(dict(id=cid, why=q['drop'])); continue
    if m['split'] == 'train': assert m['identity'] not in HELDOUT, cid
    clips.append(dict(id=cid, identity=m['identity'], split=m['split'], origin=m['origin']))
order = {'heldout': 0, 'heldout_ref': 1, 'train': 2}
clips.sort(key=lambda c: (order[c['split']], c['origin'] != 'render2real', c['id']))
man = dict(clips=clips, status='final' if '--final' in sys.argv else 'growing', heldout_identities=HELDOUT,
           counts=dict(collections.Counter(c['split'] for c in clips)), dropped=dropped,
           format='clips/<id>/{real.mp4 384x384 30fps 240f, audio.wav 16k 8.000s, audio_pad.wav 16k 10.000s (1 s before/after), lm.npz, meta.json}')
json.dump(man, open(f'{D}/manifest.json.tmp', 'w'), indent=1); os.replace(f'{D}/manifest.json.tmp', f'{D}/manifest.json')
print(man['counts'], 'dropped', len(dropped))
