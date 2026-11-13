"""Blind realism judge — Claude subscription only (`claude -p --model sonnet`, Read tool), same calling pattern as
render2real/eval/judge.py.  Per clip: 6-frame contact strips of the generated and the real video (same frame times),
under neutral random file names.  2 trials with swapped order; each trial = 2 calls:
  (a) realism: both strips rated 1-10 with render2real's realism wording ("how real each looks, 1 (obviously CG/cartoon)
      to 10 (indistinguishable from a real camera)") -> realism (generated), realism_real
  (b) forced choice: "exactly one is real camera footage, which one?" -> fooled = judge picked the generated strip
Cached per (run, clip, generated-file signature) in eval/judge_cache/<run>/<clip>.json.
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
VERSION = 1

SYSTEM = ("You are a careful visual quality rater. You compare strips of video frames. You judge only what is visible. "
          "You do not know how the images were made. Answer strictly in the requested JSON.")
REAL_SCHEMA = {"type": "object", "properties": {
    "realism_A": {"type": "integer", "minimum": 1, "maximum": 10},
    "realism_B": {"type": "integer", "minimum": 1, "maximum": 10},
    "reason": {"type": "string"}}, "required": ["realism_A", "realism_B", "reason"]}
CHOICE_SCHEMA = {"type": "object", "properties": {
    "real": {"type": "string", "enum": ["A", "B"]},
    "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    "reason": {"type": "string"}}, "required": ["real", "confidence", "reason"]}

REAL_PROMPT = """Two horizontal strips of video frames (6 frames each, left to right = time) are in your working directory:
{a} (call it strip A) and {b} (call it strip B). Use the Read tool on both files, then answer.
Each strip shows the same short moment of a person talking, from the same camera framing.
1. realism_A, realism_B: how real each looks, 1 (obviously CG/cartoon) to 10 (indistinguishable from a real camera).
2. reason: one short line.
Return only the JSON."""
CHOICE_PROMPT = """Two horizontal strips of video frames (6 frames each, left to right = time) are in your working directory:
{a} (call it strip A) and {b} (call it strip B). Use the Read tool on both files, then answer.
Each strip shows the same short moment of a person talking, from the same camera framing.
Exactly one of the two strips is real camera footage of a real person; the other one was synthesised by a computer.
1. real: which strip is the real camera footage? Forced choice, A or B.
2. confidence: 1 (pure guess) to 5 (certain).
3. reason: one short line.
Return only the JSON."""


def strip(path, out_png, shots=6, size=384):
    cap = cv2.VideoCapture(path); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0: return False
    fr = []
    for k in range(shots):
        i = int((k + 0.5) * n / shots); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
        if not ok: return False
        fr.append(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA))
    cv2.imwrite(out_png, np.hstack(fr)); return True


def call_claude(prompt, schema, cwd):
    cmd = [CLAUDE, '-p', '--model', MODEL, '--tools', 'Read', '--permission-mode', 'dontAsk', '--no-session-persistence',
           '--setting-sources', '', '--system-prompt', SYSTEM, '--output-format', 'json', '--json-schema', json.dumps(schema)]
    last = None
    for attempt in (1, 2, 3):
        try:
            r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600)
            d = json.loads(r.stdout)
            out = d.get('structured_output')
            if out is None: out = json.loads(re.search(r'\{.*\}', d.get('result') or '', re.S).group(0))
            return out
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:200]}'
            time.sleep(2 * attempt)
    return {'error': last}


def sig(p):
    st = os.stat(p); return f'{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}'


def judge_run(run, items, workers=4, seed='1', log=print):
    """items: list of (clip_id, gen_path, real_path). Returns {clip: result}. Uses/fills the per-clip cache."""
    os.makedirs(os.path.join(CACHE, run), exist_ok=True)
    results, todo = {}, []
    for cid, gp, rp in items:
        key = hashlib.md5(f'v{VERSION}|{MODEL}|{sig(gp)}|{sig(rp)}'.encode()).hexdigest()
        cp = os.path.join(CACHE, run, cid + '.json')
        if os.path.exists(cp):
            try:
                c = json.load(open(cp))
                if c.get('key') == key and not any('error' in t for t in c['trials']): results[cid] = c; continue
            except Exception: pass
        rnd = random.Random(f'{seed}-{run}-{cid}')
        d = os.path.join(WORK, f'{run}_{cid}_{key[:6]}'); os.makedirs(d, exist_ok=True)
        names = {}
        for role, p in (('gen', gp), ('real', rp)):
            nm = f'img_{rnd.randrange(10**6):06d}.png'
            while nm in names.values(): nm = f'img_{rnd.randrange(10**6):06d}.png'
            names[role] = nm
            if not strip(p, os.path.join(d, nm)): names = None; break
        if names is None: results[cid] = dict(clip=cid, error='cannot read video'); continue
        first = rnd.choice([('gen', 'real'), ('real', 'gen')])
        orders = [first, first[::-1]]
        todo.append((cid, cp, key, d, names, orders))
    calls = []
    for cid, cp, key, d, names, orders in todo:
        for k, (a, b) in enumerate(orders):
            calls.append((cid, k, 'realism', d, REAL_PROMPT.format(a=names[a], b=names[b]), REAL_SCHEMA, (a, b)))
            calls.append((cid, k, 'choice', d, CHOICE_PROMPT.format(a=names[a], b=names[b]), CHOICE_SCHEMA, (a, b)))
    log(f'judge {run}: {len(results)} cached clips, {len(calls)} calls to make')
    t0 = time.time()
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        outs = list(ex.map(lambda c: call_claude(c[4], c[5], c[3]), calls))
    log(f'judge {run}: calls done in {time.time() - t0:.0f}s')
    by = {}
    for c, o in zip(calls, outs): by.setdefault(c[0], {}).setdefault(c[1], {})[c[2]] = (c[6], o)
    for cid, cp, key, d, names, orders in todo:
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
        if len(ok) == 2:
            tmp = cp + '.tmp'; json.dump(res, open(tmp, 'w'), indent=1); os.replace(tmp, cp)
        results[cid] = res
    return results


def summarize(results):
    ok = [r for r in results.values() if 'realism' in r]
    tr = [t for r in ok for t in r['trials'] if 'error' not in t]
    if not tr: return dict(clips=0, trials=0)
    k = sum(t['fooled'] for t in tr); n = len(tr)
    s = dict(clips=len(ok), trials=n, score_1_10=round(float(np.mean([t['realism_gen'] for t in tr])), 3),
             score_real_1_10=round(float(np.mean([t['realism_real'] for t in tr])), 3),
             fooled_rate=round(k / n, 3), errors=sum('error' in t for r in results.values() for t in r.get('trials', [])))
    # with swapped order, picking the same ROLE twice = consistent; picked_as_real holds roles, so consistent = equal roles
    cons = [r['trials'][0]['picked_as_real'] == r['trials'][1]['picked_as_real'] for r in ok
            if len(r['trials']) == 2 and all('error' not in t for t in r['trials'])]
    s['order_consistency'] = round(float(np.mean(cons)), 3) if cons else None
    return s
