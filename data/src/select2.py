"""pose2real window selection for HELDOUT_REF + EXPAND. Same gates/crop recipe as render2real select.py
(frontal >= 0.9, speech >= 0.5, face box inside static square crop on >= 96% frames (crop may shift <= 0.12 S), S = 2*ear-to-ear/0.98, nose at 47% of S),
plus S >= 250 px (excludes standing full-body people; render2real's smallest kept S was 272).
Greedy per video: best-scoring windows first, starts >= MINSEP frames apart, no overlap with any render2real window of the same video.
Writes work/sel2.json: list of {id, identity, split, file_id, video, st, S, x0, y0, speech, frontal, mot, score}."""
import json, glob, os, collections, numpy as np
SEAM = __import__('os').path.expandvars('$SEAMLESS_ROOT')
T48 = __import__('os').path.expandvars('$R2R_DATA'); D = __import__('os').path.expandvars('$WORK/data')
HELDOUT = ['P0115', 'P0485', 'P0845', 'P1067', 'P0989', 'P0217', 'P0370', 'P1174']
# the 9 Seamless identities filmed standing full-body (render2real REPORT; confirmed visually 11-13 for the 6 that pass the keypoint gates)
STANDING = {'P0371', 'P0466', 'P0484', 'P0500', 'P0660', 'P0665', 'P0816', 'P0879', 'P1041'}
L = 240; MINSEP = 300; CAP = 40; SMIN = 250; INSIDE = 0.96
# existing render2real windows
old = collections.defaultdict(list); held = collections.defaultdict(list)
for c in json.load(open(f'{T48}/manifest.json'))['clips']:
    m = json.load(open(f"{T48}/pairs/{c['id']}/meta.json"))
    old[m['file_id']].append(m['src_start_frame'])
    if c['split'] == 'heldout': held[c['identity']].append((m['file_id'], m['src_start_frame']))
allw = []
for vid in sorted(glob.glob(f'{SEAM}/*/dev/video/*.mp4')):
    fid = os.path.basename(vid)[:-4]; base = os.path.dirname(os.path.dirname(vid)); pid = fid.split('_')[-1]
    if pid in STANDING: print('standing', fid); continue
    rel = pid in HELDOUT   # heldout_ref windows: relaxed gates allowed (strict ones preferred by score)
    try:
        kp = np.load(f'{base}/boxes_and_keypoints/keypoints/{fid}.npy'); vb = np.load(f'{base}/boxes_and_keypoints/is_valid_box/{fid}.npy').reshape(-1)
    except Exception as e: print('skip', fid, e); continue
    vp = f'{base}/metadata/vad/{fid}.jsonl'
    vad = [json.loads(l) for l in open(vp)] if os.path.exists(vp) else []
    T = len(kp); sp = np.zeros(T)
    for s in vad: sp[int(s['start'] * 30):int(s['end'] * 30)] = 1
    nose, le, re_ = kp[:, 0], kp[:, 3], kp[:, 4]
    ed = np.abs(le[:, 0] - re_[:, 0]); mid = (le[:, 0] + re_[:, 0]) / 2
    frontal = (le[:, 2] > 0.5) & (re_[:, 2] > 0.5) & (nose[:, 2] > 0.6) & (np.abs(nose[:, 0] - mid) / np.maximum(ed, 1) < 0.18) & (vb[:T] > 0)
    picks = []
    for st in range(30, T - L - 30, 15):
        w = slice(st, st + L)
        fr = frontal[w].mean(); s = sp[w].mean()
        strict = fr >= 0.9 and s >= 0.5
        if not strict and not (rel and fr >= 0.8 and s >= 0.25): continue
        e = np.median(ed[w][frontal[w]]); S = int(round(2.0 * e / 0.98)) // 2 * 2
        if S < SMIN or S > 1080: continue
        nx, ny = np.median(nose[w][frontal[w], 0]), np.median(nose[w][frontal[w], 1])
        fx0 = nose[w, 0] - 0.6 * ed[w]; fx1 = nose[w, 0] + 0.6 * ed[w]; fy0 = nose[w, 1] - 0.85 * ed[w]; fy1 = nose[w, 1] + 0.75 * ed[w]
        fm = frontal[w]
        # render2real placed the crop at the median nose; here the static crop may shift by up to 0.12*S from that
        # (face then sits 35-59% across) to keep a moving speaker inside. Smallest shift with the best inside fraction wins.
        best = None
        for dx in (0, -0.04, 0.04, -0.08, 0.08, -0.12, 0.12):
            for dy in (0, -0.04, 0.04, -0.08, 0.08):
                x0 = int(np.clip(nx - S / 2 + dx * S, 0, 1080 - S)); y0 = int(np.clip(ny - 0.47 * S + dy * S, 0, 1920 - S))
                ins = ((fx0 > x0) & (fx1 < x0 + S) & (fy0 > y0) & (fy1 < y0 + S))[fm].mean()
                if best is None or ins > best[0] + 1e-9: best = (ins, x0, y0, dx, dy)
        inside, x0, y0, dx, dy = best
        mot = np.std(nose[w][frontal[w], :2], 0).max() / e
        strict = strict and inside >= INSIDE
        if inside < (0.92 if rel else INSIDE) or y0 == 0 and ny - 0.47 * S < -0.05 * S: continue
        if any(abs(st - o) < L + 15 for o in old.get(fid, [])): continue   # no overlap with render2real windows
        picks.append(dict(score=float(fr + s - mot - (0 if strict else 1)), strict=bool(strict), st=st, S=S, x0=x0, y0=y0, speech=round(float(s), 2), frontal=round(float(fr), 2), mot=round(float(mot), 3), inside=round(float(inside), 3), shift=[dx, dy]))
    picks.sort(key=lambda p: -p['score']); chosen = []
    for p in picks:
        if all(abs(p['st'] - c['st']) >= MINSEP for c in chosen): chosen.append(p)
    chosen.sort(key=lambda p: p['st'])
    for k, c in enumerate(chosen): allw.append(dict(file_id=fid, video=vid, identity=pid, **c))
    print(fid, T, 'passing', len(picks), 'chosen', len(chosen), flush=True)
# assignment
byid = collections.defaultdict(list)
for w in allw: byid[w['identity']].append(w)
sel = []
for pid, ws in sorted(byid.items()):
    if pid in HELDOUT:
        hv = held[pid]
        ok = [w for w in ws if all(w['file_id'] != f or abs(w['st'] - s) >= 1140 for f, s in hv)]   # >= 30 s gap between window edges
        hfiles = {f for f, _ in hv}
        ok.sort(key=lambda w: (w['file_id'] in hfiles, -w['score']))   # other videos first
        got = []
        for sep, n in ((900, 4), (300, 2)):   # refs >= 30 s apart within a video; if that leaves < 2, allow 10 s (non-overlapping)
            for w in ok:
                if len(got) >= n: break
                if w in got or any(g['file_id'] == w['file_id'] and abs(g['st'] - w['st']) < sep for g in got): continue
                got.append(w)
        for w in got: w['split'] = 'heldout_ref'
        sel += got
    else:
        # round robin over videos (best windows first within each video), cap CAP per identity
        perv = collections.defaultdict(list)
        for w in sorted(ws, key=lambda w: -w['score']): perv[w['file_id']].append(w)
        got = []; qs = [perv[k] for k in sorted(perv)]
        while len(got) < CAP and any(qs):
            for q in qs:
                if q and len(got) < CAP: got.append(q.pop(0))
        for w in got: w['split'] = 'train'
        sel += got
# ids: X = expand train, R = heldout_ref; order: refs first, then train interleaved by identity so early clips span identities
refs = [w for w in sel if w['split'] == 'heldout_ref']; trs = [w for w in sel if w['split'] == 'train']
for i, w in enumerate(refs): w['id'] = f"R{i:03d}_{w['identity']}"
bi = collections.defaultdict(list)
for w in trs: bi[w['identity']].append(w)
order = []; qs = [bi[k] for k in sorted(bi)]
while any(qs):
    for q in qs:
        if q: order.append(q.pop(0))
for i, w in enumerate(order): w['id'] = f"X{i:04d}_{w['identity']}"
json.dump(refs + order, open(f'{D}/work/sel2.json', 'w'), indent=1)
print('windows', len(allw), 'heldout_ref', len(refs), dict(collections.Counter(w['identity'] for w in refs)))
print('train', len(order), 'identities', len(bi), dict(sorted(collections.Counter(w['identity'] for w in order).items())))
