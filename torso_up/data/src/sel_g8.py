"""gesture8 candidates: per heldout identity, windows with speech >= .5, frontal >= .85, >= 1140 frames (30 s gap + window)
from every same12 window of the same video; prefer >= 1140 from heldout_ref windows too. score = speed/0.01 + raised.
Writes work/g8_cands.json (top 4 per identity, >= 600 frames apart)."""
import json, collections
D = __import__('os').path.expandvars('$WORK/data/work')
H = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
c = json.load(open(f'{D}/cands.json')); pc = json.load(open(f'{D}/prior_crops.json'))
same = collections.defaultdict(list); refs = collections.defaultdict(list)
for k, v in pc.items():
    if v['split'] == 'heldout': same[v['src_video']].append(v['st'])
    if v['split'] == 'heldout_ref': refs[v['src_video']].append(v['st'])
out = {}
for h in H:
    ws = [w for w in c if w['identity'] == h and w['speech'] >= 0.5 and w['frontal'] >= 0.85 and w['handvis'] >= 0.9
          and all(abs(w['st'] - s) >= 1140 for s in same[w['video']])]
    for w in ws:
        w['ref_conflict'] = [s for s in refs[w['video']] if abs(w['st'] - s) < 1140]
        w['score'] = round(w['speed'] / 0.01 + w['raised'] - 0.3 * bool(w['ref_conflict']), 3)
    ws.sort(key=lambda w: -w['score']); got = []
    for w in ws:
        if all(g['video'] != w['video'] or abs(g['st'] - w['st']) >= 600 for g in got): got.append(w)
        if len(got) >= 4: break
    out[h] = got
    print(h, len(ws), [(g['file_id'][-16:], g['st'], g['score'], g['speed'], g['raised'], g['speech'], len(g['ref_conflict'])) for g in got])
json.dump(out, open(f'{D}/g8_cands.json', 'w'), indent=0)
# crops json for sheet_raw
cr = {f"{h}_{i}": dict(src_video=g['video'], st=g['st'], crop=g['crop']) for h, gs in out.items() for i, g in enumerate(gs)}
json.dump(cr, open(f'{D}/g8_sheet.json', 'w'))
print(' '.join(cr))
