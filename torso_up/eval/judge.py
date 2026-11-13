"""Blind realism judges — Claude subscription only (`claude -p --model sonnet --tools Read`), call pattern of pose2real/eval/judge.py.

1) Whole-frame judge (protocol of pose2real, new aspect): 6 frames (same frame times) of the generated and of the real video,
   each laid out as a 3x2 grid of full-resolution 512x640 frames (1536x1280, read left->right, top->bottom = time) under
   neutral random file names. 2 trials per clip with swapped order; per trial 2 calls:
     (a) realism: both grids rated 1-10 ("1 = obviously CG/cartoon, 10 = indistinguishable from a real camera")
     (b) forced choice: "exactly one is real camera footage, which one?"
   -> realism (generated), realism_real, fooled_rate.  80 calls for 20 clips.
2) Hands judge (new): clips whose conditioning has >= 12 hand frames. 6 frames evenly spaced over the frames where the
   conditioning has a hand; per frame a tile with the LEFT hand crop above the RIGHT hand crop (grey if the conditioning has
   no such hand that frame). Crop box = square around the CONDITIONING hand landmarks (centre of the 21-point bbox; side =
   1.5 x the clip's median hand extent (max side of the 21-point bbox), clamped to 48..192 px, same side for the whole clip),
   the SAME box for generated and real, upscaled with Lanczos to min(2 x side, 256) px (2x for crops <= 128 px; a typical
   torso-up hand is ~80-110 px across -> crop ~130-165 px -> ~1.6-2x). 1 call per clip (order random) rating both strips:
   "how real and anatomically correct do the hands look, 1-10" -> hands_score (generated), hands_score_real.
Cached per (run, clip, both file signatures, model, version) in eval/judge_cache/<run>/<clip>.json and <clip>.hands.json.
"""
import json, os, random, re, subprocess, time, hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CLAUDE = os.environ.get('JUDGE_CLAUDE', '/usr/local/bin/claude')
MODEL = os.environ.get('JUDGE_MODEL', 'sonnet')
WORK = os.path.join(HERE, 'judge_work')
CACHE = os.path.join(HERE, 'judge_cache')
VERSION = 2
HANDS_VERSION = 2
MIN_HAND_FRAMES = 12
TILE_MAX = 256   # crop upscale = min(2, 256/side): exactly 2x for crops <= 128 px

SYSTEM = ("You are a careful visual quality rater. You compare images of video frames. You judge only what is visible. "
          "You do not know how the images were made. Answer strictly in the requested JSON.")
REAL_SCHEMA = {"type": "object", "properties": {
    "realism_A": {"type": "integer", "minimum": 1, "maximum": 10},
    "realism_B": {"type": "integer", "minimum": 1, "maximum": 10},
    "reason": {"type": "string"}}, "required": ["realism_A", "realism_B", "reason"]}
CHOICE_SCHEMA = {"type": "object", "properties": {
    "real": {"type": "string", "enum": ["A", "B"]},
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    "reason": {"type": "string"}}, "required": ["real", "confidence", "reason"]}
HANDS_SCHEMA = {"type": "object", "properties": {
    "hands_A": {"type": "integer", "minimum": 1, "maximum": 10},
    "hands_B": {"type": "integer", "minimum": 1, "maximum": 10},
    "reason": {"type": "string"}}, "required": ["hands_A", "hands_B", "reason"]}

GRID = """Two images are in your working directory: {a} (call it A) and {b} (call it B). Use the Read tool on both files, then answer.
Each image is a grid of 6 video frames (read left to right, top to bottom = time) of the same short moment of a person talking,
from the same camera framing (head, shoulders, arms and hands)."""
REAL_PROMPT = GRID + """
1. realism_A, realism_B: how real each looks, 1 (obviously CG/cartoon) to 10 (indistinguishable from a real camera).
2. reason: one short line.
Return only the JSON."""
CHOICE_PROMPT = GRID + """
Exactly one of the two is real camera footage of a real person; the other one was synthesised by a computer.
1. real: which one is the real camera footage? Forced choice, A or B.
2. confidence: 1 (pure guess) to 5 (certain).
3. reason: one short line.
Return only the JSON."""
HANDS_PROMPT = """Two images are in your working directory: {a} (call it A) and {b} (call it B). Use the Read tool on both files, then answer.
Each image is a strip of 6 close-up crops (left to right = time) around the hands of a person talking; in each column the
person's left hand is on top and the right hand below (a plain grey square means that hand is not shown at that moment).
1. hands_A, hands_B: how real and anatomically correct do the hands look, 1 (malformed, melted, wrong number or shape of
   fingers, obviously synthetic) to 10 (indistinguishable from real camera footage of real hands).
2. reason: one short line.
Return only the JSON."""


def read_frames_at(path, idx):
    cap = cv2.VideoCapture(path); out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i)); ok, f = cap.read()
        if not ok: cap.release(); return None
        out.append(f)
    cap.release(); return out


def nframes(path):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release(); return n


def grid(path, out_png, shots=6, size=(512, 640), cols=3):
    n = nframes(path)
    if n <= 0: return False
    fr = read_frames_at(path, [int((k + 0.5) * n / shots) for k in range(shots)])
    if fr is None: return False
    fr = [f if f.shape[1::-1] == size else cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in fr]
    rows = [np.hstack(fr[i:i + cols]) for i in range(0, shots, cols)]
    cv2.imwrite(out_png, np.vstack(rows)); return True


def hand_plan(lm_path, W, H, shots=6):
    """frames + per-slot boxes from the CONDITIONING hands; None if too few hand frames."""
    z = np.load(lm_path)
    if 'hands' not in z.files: return None
    hands = z['hands'][..., :2] * [W, H]; ok = z['hands_ok'].astype(bool)
    anyh = np.where(ok.any(1))[0]
    if len(anyh) < MIN_HAND_FRAMES: return None
    sel = anyh[[int((k + 0.5) * len(anyh) / shots) for k in range(shots)]]
    ext = [np.max(hands[t, s].max(0) - hands[t, s].min(0)) for t in range(len(ok)) for s in (0, 1) if ok[t, s]]
    side = int(np.clip(1.5 * np.median(ext), 48, 192))
    boxes = []
    for t in sel:
        bt = []
        for s in (0, 1):
            if ok[t, s]:
                c = (hands[t, s].max(0) + hands[t, s].min(0)) / 2
                bt.append((int(round(c[0] - side / 2)), int(round(c[1] - side / 2))))
            else: bt.append(None)
        boxes.append(bt)
    return dict(frames=[int(t) for t in sel], side=side, boxes=boxes)


def hand_strip(path, plan, out_png, W=512, H=640):
    fr = read_frames_at(path, plan['frames'])
    if fr is None: return False
    side = plan['side']; tile = min(2 * side, TILE_MAX); cols = []
    for f, bt in zip(fr, plan['boxes']):
        if f.shape[1] != W or f.shape[0] != H: f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        pad = cv2.copyMakeBorder(f, side, side, side, side, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        col = []
        for b in bt:
            if b is None: col.append(np.full((tile, tile, 3), 128, np.uint8)); continue
            x, y = b[0] + side, b[1] + side
            col.append(cv2.resize(pad[y:y + side, x:x + side], (tile, tile), interpolation=cv2.INTER_LANCZOS4))
        cols.append(np.vstack(col))
    sep = np.full((2 * tile, 4, 3), 255, np.uint8)
    out = []
    for c in cols: out += [c, sep]
    cv2.imwrite(out_png, np.hstack(out[:-1])); return True


def call_claude(prompt, schema, cwd):
    cmd = [CLAUDE, '-p', '--model', MODEL, '--tools', 'Read', '--permission-mode', 'dontAsk', '--no-session-persistence',
           '--setting-sources', '', '--system-prompt', SYSTEM, '--output-format', 'json', '--json-schema', json.dumps(schema)]
    env = {k: v for k, v in os.environ.items() if k not in ('ANTHROPIC_API_KEY', 'OPENAI_API_KEY')}  # subscription only
    last = None
    for attempt in (1, 2, 3):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600, env=env)
            d = json.loads(r.stdout)
            out = d.get('structured_output')
            if out is None: out = json.loads(re.search(r'\{.*\}', d.get('result') or '', re.S).group(0))
            for k in schema['required']:
                if k not in out: raise ValueError(f'missing {k}')
            return out
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(2 * attempt)
    return {'error': last}


def sig(p):
    st = os.stat(p); return f'{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}'


def _names(rnd, roles):
    names = {}
    for role in roles:
        nm = f'img_{rnd.randrange(10**6):06d}.png'
        while nm in names.values(): nm = f'img_{rnd.randrange(10**6):06d}.png'
        names[role] = nm
    return names


def _load_cache(cp, key):
    if not os.path.exists(cp): return None
    try:
        c = json.load(open(cp))
        if c.get('key') == key and not any('error' in t for t in c.get('trials', [])): return c
    except Exception: pass
    return None


def _save(cp, res):
    tmp = cp + f'.{os.getpid()}.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, cp)


def judge_run(run, items, workers=4, seed='1', log=print, hands=True, W=512, H=640):
    """items: list of (clip_id, gen_path, real_path, cond_lm_path). Returns {clip: result} (result['hands'] = hands judge)."""
    os.makedirs(os.path.join(CACHE, run), exist_ok=True)
    results, todo, htodo, calls = {}, [], [], []
    for cid, gp, rp, lmp in items:
        rnd = random.Random(f'{seed}-{run}-{cid}')
        key = hashlib.md5(f'v{VERSION}|{MODEL}|{sig(gp)}|{sig(rp)}'.encode()).hexdigest()
        cp = os.path.join(CACHE, run, cid + '.json')
        c = _load_cache(cp, key)
        if c: results[cid] = c
        else:
            d = os.path.join(WORK, f'{run}_{cid}_{key[:6]}'); os.makedirs(d, exist_ok=True)
            names = _names(rnd, ('gen', 'real'))
            if not all(grid(p, os.path.join(d, names[r])) for r, p in (('gen', gp), ('real', rp))):
                results[cid] = dict(clip=cid, error='cannot read video')
            else:
                first = rnd.choice([('gen', 'real'), ('real', 'gen')])
                orders = [first, first[::-1]]
                todo.append((cid, cp, key, names, orders))
                for k, (a, b) in enumerate(orders):
                    calls.append((cid, k, 'realism', d, REAL_PROMPT.format(a=names[a], b=names[b]), REAL_SCHEMA, (a, b)))
                    calls.append((cid, k, 'choice', d, CHOICE_PROMPT.format(a=names[a], b=names[b]), CHOICE_SCHEMA, (a, b)))
        if not hands or not lmp: continue
        hrnd = random.Random(f'{seed}-{run}-{cid}-hands')
        hkey = hashlib.md5(f'h{HANDS_VERSION}|{MODEL}|{sig(gp)}|{sig(rp)}|{sig(lmp)}'.encode()).hexdigest()
        hcp = os.path.join(CACHE, run, cid + '.hands.json')
        hc = _load_cache(hcp, hkey)
        if hc: results.setdefault(cid, dict(clip=cid))['hands'] = hc; continue
        plan = hand_plan(lmp, W, H)
        if plan is None:
            results.setdefault(cid, dict(clip=cid))['hands'] = dict(skipped=f'< {MIN_HAND_FRAMES} conditioning hand frames'); continue
        d = os.path.join(WORK, f'{run}_{cid}_hands_{hkey[:6]}'); os.makedirs(d, exist_ok=True)
        names = _names(hrnd, ('gen', 'real'))
        if not all(hand_strip(p, plan, os.path.join(d, names[r]), W, H) for r, p in (('gen', gp), ('real', rp))):
            results.setdefault(cid, dict(clip=cid))['hands'] = dict(error='cannot read video'); continue
        a, b = hrnd.choice([('gen', 'real'), ('real', 'gen')])
        htodo.append((cid, hcp, hkey, names, (a, b), plan))
        calls.append((cid, 0, 'hands', d, HANDS_PROMPT.format(a=names[a], b=names[b]), HANDS_SCHEMA, (a, b)))
    log(f'judge {run}: {len(items) - len(todo)} clips cached (whole frame), {len(calls)} calls to make')
    t0 = time.time()
    with ThreadPoolExecutor(max(1, min(workers, 4))) as ex:
        outs = list(ex.map(lambda c: call_claude(c[4], c[5], c[3]), calls))
    if calls: log(f'judge {run}: {len(calls)} calls done in {time.time() - t0:.0f}s')
    by = {}
    for c, o in zip(calls, outs): by.setdefault(c[0], {}).setdefault(c[1], {})[c[2]] = (c[6], o)
    for cid, cp, key, names, orders in todo:
        trials = []
        for k in range(2):
            (a, b), ro = by[cid][k]['realism']; _, co = by[cid][k]['choice']
            t = dict(order=f'{a}/{b}')
            if 'error' in ro or 'error' in co: t['error'] = ro.get('error') or co.get('error'); trials.append(t); continue
            t['realism_gen'] = ro['realism_A' if a == 'gen' else 'realism_B']
            t['realism_real'] = ro['realism_A' if a == 'real' else 'realism_B']
            picked = {'A': a, 'B': b}[co['real']]
            t['picked_as_real'] = picked; t['fooled'] = picked == 'gen'; t['confidence'] = co['confidence']
            t['reason_realism'] = ro['reason']; t['reason_choice'] = co['reason']
            trials.append(t)
        res = dict(clip=cid, key=key, model=MODEL, strips=names, trials=trials)
        ok = [t for t in trials if 'error' not in t]
        if ok:
            res['realism'] = float(np.mean([t['realism_gen'] for t in ok]))
            res['realism_real'] = float(np.mean([t['realism_real'] for t in ok]))
            res['fooled_rate'] = float(np.mean([t['fooled'] for t in ok]))
        if len(ok) == 2: _save(cp, res)
        h = results.get(cid, {}).get('hands')
        results[cid] = res
        if h is not None: res['hands'] = h
    for cid, hcp, hkey, names, (a, b), plan in htodo:
        _, ho = by[cid][0]['hands']
        t = dict(order=f'{a}/{b}')
        if 'error' in ho: t['error'] = ho['error']
        else:
            t['hands_gen'] = ho['hands_A' if a == 'gen' else 'hands_B']; t['hands_real'] = ho['hands_A' if a == 'real' else 'hands_B']
            t['reason'] = ho['reason']
        hres = dict(clip=cid, key=hkey, model=MODEL, strips=names, plan=plan, trials=[t])
        if 'error' not in t:
            hres['hands_score'] = float(t['hands_gen']); hres['hands_score_real'] = float(t['hands_real']); _save(hcp, hres)
        results.setdefault(cid, dict(clip=cid))['hands'] = hres
    return results


def summarize(results):
    ok = [r for r in results.values() if 'realism' in r]
    tr = [t for r in ok for t in r['trials'] if 'error' not in t]
    s = dict(clips=len(ok), trials=len(tr))
    if tr:
        k = sum(t['fooled'] for t in tr); n = len(tr)
        s.update(score_1_10=round(float(np.mean([t['realism_gen'] for t in tr])), 3),
                 score_real_1_10=round(float(np.mean([t['realism_real'] for t in tr])), 3), fooled_rate=round(k / n, 3))
        cons = [r['trials'][0]['picked_as_real'] == r['trials'][1]['picked_as_real'] for r in ok
                if len(r['trials']) == 2 and all('error' not in t for t in r['trials'])]
        s['order_consistency'] = round(float(np.mean(cons)), 3) if cons else None
    s['errors'] = sum('error' in t for r in results.values() for t in r.get('trials', []))
    hs = [r['hands'] for r in results.values() if 'hands_score' in r.get('hands', {})]
    s['hands_clips'] = len(hs)
    if hs:
        s['hands_score_1_10'] = round(float(np.mean([h['hands_score'] for h in hs])), 3)
        s['hands_score_real_1_10'] = round(float(np.mean([h['hands_score_real'] for h in hs])), 3)
    s['hands_errors'] = sum('error' in t for r in results.values() for t in r.get('hands', {}).get('trials', []))
    return s
