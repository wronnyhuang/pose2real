"""Build work/plan.json: every window to cut, in processing order (heldout 20 first).
  same12      : PRIOR heldout windows, same ids, re-cropped
  gesture8    : G0k_<pid>, hand-picked from work/g8_cands.json after looking at work/sheet_g8c.jpg (see G8 below)
  heldout_ref : PRIOR refs re-cropped; a ref within 1140 frames (30 s + window) of a gesture8 window of the same video is replaced
                by a new window of that identity (R1xx ids), >= 1140 from every same12/gesture8 window, >= 300 from other refs
  train/prior : PRIOR's 853 train windows re-cropped (same ids)
  train/gesture: N<nnn>_<pid> new gesture-rich windows of non-heldout identities (standing identities allowed if face_px >= 62)"""
import json, collections, sys
D = __import__('os').path.expandvars('$WORK/data/work')
P = __import__('os').path.expandvars('$WORK/data')
H = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
N_NEW = 250; CAP_NEW = 14; FACE_MIN = 62     # ear-to-ear px in 512x640 (MediaPipe face width ~0.9 of it -> >= ~55 px)
G8 = [('P0115', 'I00000379_P0115', 2160), ('P0485', None, 1980), ('P0845', 'I00000538_P0845', 1125),
      ('P1067', None, 1335), ('P0989', None, 6975), ('P0217', 'I00000385_P0217', 645), ('P0370', None, 4530), ('P1174', None, 30)]
cands = json.load(open(f'{D}/cands.json'))
idx = {(w['video'], w['st']): w for w in cands}
prior = json.load(open(f'{D}/prior_crops.json'))
plan = []
def add(cid, w, split, set_, origin, prior_id=None):
    plan.append(dict(id=cid, identity=w['identity'], split=split, set=set_, origin=origin, video=w['video'], file_id=w['file_id'], st=w['st'],
                     crop=w['crop'], speech=w.get('speech'), frontal=w.get('frontal'), raised=w.get('raised'), speed=w.get('speed'),
                     face_px=w.get('face_px'), prior_id=prior_id))
def win(video, st):
    w = idx.get((video, st))
    assert w is not None, (video, st)
    return w
# same12
for k, v in sorted(prior.items()):
    if v['split'] == 'heldout': add(k, win(v['src_video'], v['st']), 'heldout', 'same12', 'prior', k)
# gesture8
g8 = []
for i, (pid, fid, st) in enumerate(G8):
    ws = [w for w in cands if w['identity'] == pid and w['st'] == st and (fid is None or fid in w['file_id'])]
    assert len(ws) == 1, (pid, fid, st, len(ws))
    add(f'G{i:02d}_{pid}', ws[0], 'heldout', 'gesture8', 'new'); g8.append(ws[0])
held = [(p['video'], p['st']) for p in plan]
# heldout_ref
refs = [(k, v) for k, v in sorted(prior.items()) if v['split'] == 'heldout_ref']
keep, replace = [], []
for k, v in refs:
    if any(g['video'] == v['src_video'] and abs(g['st'] - v['st']) < 1140 for g in g8): replace.append((k, v))
    else: keep.append((k, v))
for k, v in keep: add(k, win(v['src_video'], v['st']), 'heldout_ref', 'prior', 'prior', k)
refpos = [(v['src_video'], v['st']) for k, v in keep]
nr = 100
for k, v in replace:
    pid = v['identity']
    def ok(w, sep):
        return (all(vv != w['video'] or abs(s - w['st']) >= 1140 for vv, s in held)
                and all(vv != w['video'] or abs(s - w['st']) >= sep for vv, s in refpos))
    best = None
    for sp_min, fr_min, sep in ((0.5, 0.9, 900), (0.5, 0.85, 300), (0.35, 0.8, 300), (0.25, 0.7, 255)):
        ws = [w for w in cands if w['identity'] == pid and w['speech'] >= sp_min and w['frontal'] >= fr_min and ok(w, sep)]
        if ws:
            best = max(ws, key=lambda w: (w['video'] != v['src_video'], w['frontal'] + w['speech'])); break
    if best is None:
        print('ref', k, 'dropped: no replacement window'); continue
    cid = f'R{nr}_{pid}'; nr += 1
    add(cid, best, 'heldout_ref', 'replacement', 'new', None); refpos.append((best['video'], best['st']))
    plan[-1]['replaces'] = k
    print('ref', k, 'replaced by', cid, best['file_id'], best['st'], best['speech'], best['frontal'])
# train prior
for k, v in sorted(prior.items()):
    if v['split'] == 'train': add(k, win(v['src_video'], v['st']), 'train', 'prior', 'prior', k)
# train gesture (new)
used = collections.defaultdict(list)
for k, v in prior.items(): used[v['src_video']].append(v['st'])
pool = [w for w in cands if w['identity'] not in H and w['speech'] >= 0.5 and w['frontal'] >= 0.9 and w['handvis'] >= 0.9
        and w['face_px'] >= FACE_MIN and w['crop']['scale'] <= 1.4 and all(abs(w['st'] - s) >= 255 for s in used[w['video']])]
for w in pool: w['gscore'] = w['speed'] / 0.01 + w['raised']
pool = [w for w in pool if w['gscore'] >= 2.0]
byid = collections.defaultdict(list)
for w in sorted(pool, key=lambda w: -w['gscore']): byid[w['identity']].append(w)
chosen = collections.defaultdict(list)
def free(w, got):
    return all(g['video'] != w['video'] or abs(g['st'] - w['st']) >= 300 for g in got)
rounds = 0
while sum(len(v) for v in chosen.values()) < N_NEW and rounds < CAP_NEW:
    rounds += 1
    for pid in sorted(byid, key=lambda p: -byid[p][0]['gscore'] if byid[p] else 0):
        if sum(len(v) for v in chosen.values()) >= N_NEW: break
        while byid[pid]:
            w = byid[pid].pop(0)
            if free(w, chosen[pid]): chosen[pid].append(w); break
n = 0
for pid in sorted(chosen):
    for w in chosen[pid]:
        add(f'N{n:03d}_{pid}', w, 'train', 'gesture', 'new'); n += 1
json.dump(plan, open(f'{D}/plan.json', 'w'), indent=0)
c = collections.Counter((p['split'], p['set']) for p in plan)
print(dict(c), 'total', len(plan))
print('new train identities', len(chosen), {k: len(v) for k, v in sorted(chosen.items())})
assert not any(p['split'] == 'train' and p['identity'] in H for p in plan)
