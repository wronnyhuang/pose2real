#!/usr/bin/env python3
"""torso_up eval harness (CPU only), torso-up 512x640.
    python eval/run_eval.py --run <run> [--dir <folder with <clip_id>.mp4>] [--no-judge] [--no-hands-judge] [--workers N]
                            [--sync-box mp|s3fd] [--sync-interp lanczos|linear] [--clips ...] [--out file]
default dir = ROOT/runs/<run>/out ; writes eval/results/<run>.json atomically.  See eval/README.md.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRIOR = __import__('os').path.expandvars('$WORK')
CACHE = os.path.join(HERE, 'cache')
RESULTS = os.path.join(HERE, 'results')
N_FRAMES, W, H, FPS = 240, 512, 640, 30
MIN_FRAMES = 230               # shorter outputs are a per-clip error; 230..239 are padded (last frame cloned) and flagged
SYNC_INTERP = 'linear'         # = reference pipeline / pose2real; lanczos tested in validate.py: no gain (README)
SETS = ('same12', 'gesture8')


def heldout_done(): return os.path.exists(os.path.join(ROOT, 'data', 'HELDOUT_DONE'))


def heldout():
    """[(id, set)] of the test set. Dev fallback before HELDOUT_DONE: PRIOR's 12 bust heldout clips (set same12)."""
    if heldout_done():
        mp_ = os.path.join(ROOT, 'data', 'manifest.json')
        if os.path.exists(mp_):
            cl = json.load(open(mp_))['clips']
        else:   # manifest is written with the full dataset; until then the clips' meta.json carry split/set
            import glob
            cl = [json.load(open(p)) for p in glob.glob(os.path.join(ROOT, 'data', 'clips', '*', 'meta.json'))]
        return sorted(((c['id'], c.get('set')) for c in cl if c.get('split') == 'heldout'), key=lambda x: (x[1] != 'same12', x[0]))
    m = json.load(open(os.path.join(PRIOR, 'data', 'manifest.json')))
    return sorted((c['id'], 'same12') for c in m['clips'] if c.get('split') == 'heldout')


def inputs(cid):
    base = os.path.join(ROOT, 'data', 'clips', cid) if heldout_done() else os.path.join(PRIOR, 'data', 'clips', cid)
    return dict(real=os.path.join(base, 'real.mp4'), audio=os.path.join(base, 'audio.wav'),
                audio_pad=os.path.join(base, 'audio_pad.wav'), lm=os.path.join(base, 'lm.npz'),
                source='ROOT/data/clips' if heldout_done() else 'DEV fallback: pose2real bust clips (384x384)')


def sig(p):
    st = os.stat(p); return f'{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}'


def lmk_path():
    p = os.path.join(ROOT, 'common', 'lmk.py')
    return p if os.path.exists(os.path.join(ROOT, 'common', 'LMK_READY')) else os.path.join(PRIOR, 'common', 'lmk.py')


_LMK = None
def lmk_mod():
    global _LMK
    if _LMK is None:
        import importlib.util
        p = lmk_path()
        spec = importlib.util.spec_from_file_location('torso_up_lmk', p); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _LMK = (m, hashlib.md5(open(p, 'rb').read()).hexdigest())   # signature = lmk.py CONTENT (+ path below)
    return _LMK


def cached_lmk(video, key_sig=None):
    """lmk.py output for a video file, cached by (file signature, lmk.py content hash)."""
    m, lsig = lmk_mod()
    key = hashlib.md5(f'{key_sig or sig(video)}|{lmk_path()}|{lsig}'.encode()).hexdigest()
    p = os.path.join(CACHE, 'lmk', key + '.npz')
    if os.path.exists(p):
        try:
            z = np.load(p); return {k: z[k] for k in z.files}, True
        except Exception: pass
    d = m.extract(video)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + f'.{os.getpid()}.tmp.npz'; np.savez(tmp, **d); os.replace(tmp, p)
    return d, False


def probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    r = dict(w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
             fps=round(cap.get(cv2.CAP_PROP_FPS), 3), n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release(); return r


def count_frames(path):
    import cv2
    cap = cv2.VideoCapture(path); n = 0
    while cap.grab(): n += 1
    cap.release(); return n


def normalize(path, tmpdir, w, h):
    """ensure w x h / 30 fps / 240 frames. Returns (path to use, note or None); raises ValueError for unusable videos."""
    p = probe(path)
    if p['w'] <= 0 or p['h'] <= 0: raise ValueError(f'unreadable video {p}')
    dur_frames = p['n'] * FPS / p['fps'] if p['fps'] > 0 else 0
    if p['w'] == w and p['h'] == h and abs(p['fps'] - FPS) < 0.01 and p['n'] == N_FRAMES:
        n = count_frames(path)
        if n == N_FRAMES: return path, None
        p['decoded'] = n; dur_frames = n
    if dur_frames < MIN_FRAMES - 0.5: raise ValueError(f'short video: {p} (need >= {MIN_FRAMES} frames at 30 fps)')
    out = os.path.join(tmpdir, 'norm.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-an', '-vf',
                    f'scale={w}:{h}:flags=area,fps={FPS},tpad=stop_mode=clone:stop=-1', '-frames:v', str(N_FRAMES),
                    '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', out], check=True)
    return out, f'normalized from {p}'


def load_cond(inp):
    z = np.load(inp['lm']); return {k: z[k] for k in z.files}


def eval_clip(job):
    cid, gen, box_mode, interp = job
    import cv2
    cv2.setNumThreads(1)
    sys.path.insert(0, HERE)
    import sync, lm_metrics
    t0 = time.time(); res = dict(clip=cid)
    try:
        inp = inputs(cid); res['inputs'] = inp['source']
        if not os.path.exists(gen): return dict(clip=cid, error=f'missing output {gen}', secs=0)
        if os.path.getsize(gen) == 0: return dict(clip=cid, error=f'empty output {gen}', secs=0)
        rp = probe(inp['real']); w, h = rp['w'], rp['h']
        with tempfile.TemporaryDirectory(prefix=f'ev_{cid}_', dir=os.environ.get('EVAL_TMP')) as td:
            v, note = normalize(gen, td, w, h)
            if note: res['normalized'] = note
            # ---- landmarks (generated re-tracked with the same lmk.py; cache keyed by the ORIGINAL file + 'norm')
            g, hit = cached_lmk(v) if v == gen else cached_lmk(v, key_sig=f'norm{w}x{h}|{sig(gen)}')
            cond = load_cond(inp)
            res['landmarks'] = lm_metrics.compare(cond, g, w, h)
            res['lmk_cached'] = hit; res['lmk'] = os.path.relpath(lmk_path(), os.path.dirname(ROOT))
            # ---- syncnet (cached per video + audio + box mode + interp)
            skey = hashlib.md5(f'v3|{sig(gen)}|{sig(inp["audio"])}|{box_mode}|{interp}|{w}x{h}'.encode()).hexdigest()
            sp = os.path.join(CACHE, 'sync', skey + '.json')
            if os.path.exists(sp):
                res['syncnet'] = json.load(open(sp)); res['sync_cached'] = True
            else:
                mkv, (sr, audio) = sync.mux_and_extract(v, inp['audio'], td)
                frames, fps = sync.read_frames(mkv)
                if box_mode == 's3fd':
                    idx = sync.resample_idx(len(frames), fps)
                    b25 = sync.s3fd_boxes(frames[idx], scale=1.0); c = sync.crops(frames[idx], b25, interp)
                    s = sync.score(c, audio, sr) if c is not None else dict(error='no face box')
                    s['box_det_rate'] = round(float(np.mean(~np.isnan(b25[:, 0]))), 4)
                else:
                    boxes = sync.mp_boxes(g['face'], g['face_ok'].astype(bool), w, h)
                    s = sync.sync_for(frames, fps, boxes, audio, sr, interp=interp)
                s['box'] = box_mode
                if 'error' not in s:
                    os.makedirs(os.path.dirname(sp), exist_ok=True)
                    tmp = sp + f'.{os.getpid()}.tmp'; json.dump(s, open(tmp, 'w')); os.replace(tmp, sp)
                res['syncnet'] = s; res['sync_cached'] = False
    except Exception as e:
        import traceback
        res['error'] = f'{type(e).__name__}: {e}'; res['trace'] = traceback.format_exc()[-1500:]
    res['secs'] = round(time.time() - t0, 1)
    return res


def mean_of(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.mean(v)), 4) if v else None


LM_KEYS = ('face_nme', 'mouth_nme', 'mouth_open_corr', 'det_rate', 'face_nme_pen', 'mouth_nme_pen', 'pose_err', 'pose_err_pen',
           'pose_det_rate', 'shoulder_err', 'elbow_err', 'wrist_err', 'arm_err')


def aggregate(clips, judge_res=None):
    import lm_metrics, judge
    ok = [c for c in clips if 'error' not in c]
    oks = [c for c in ok if 'error' not in c.get('syncnet', {'error': 1})]
    sy = lambda k: mean_of([c['syncnet'].get(k) for c in oks])
    m = dict(n_clips=len(ok), n_failed=len(clips) - len(ok),
             syncnet=dict(lse_c=sy('lse_c'), lse_d=sy('lse_d'), offset_frames=sy('offset_frames'),
                          abs_offset_frames=mean_of([abs(c['syncnet']['offset_frames']) for c in oks]), lse_d_at0=sy('lse_d_at0'),
                          n=len(oks)),
             landmarks={k: mean_of([c['landmarks'].get(k) for c in ok]) for k in LM_KEYS},
             hands=lm_metrics.hand_run_mean([c['landmarks'] for c in ok]))
    if judge_res is not None:
        sub = {c['clip']: judge_res[c['clip']] for c in clips if c['clip'] in judge_res}
        m['realism'] = judge.summarize(sub); m['realism']['model'] = judge.MODEL
    else: m['realism'] = None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--dir')
    ap.add_argument('--no-judge', action='store_true'); ap.add_argument('--no-hands-judge', action='store_true')
    ap.add_argument('--workers', type=int, default=12); ap.add_argument('--judge-workers', type=int, default=4)
    ap.add_argument('--sync-box', default='mp', choices=['mp', 's3fd'])
    ap.add_argument('--sync-interp', default=SYNC_INTERP, choices=['linear', 'lanczos', 'cubic', 'area'])
    ap.add_argument('--clips', nargs='*'); ap.add_argument('--out')
    a = ap.parse_args()
    try: os.nice(10)
    except Exception: pass
    os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('SYNC_THREADS', '2')
    d = a.dir or os.path.join(ROOT, 'runs', a.run, 'out')
    hs = heldout(); setof = dict(hs)
    ids = a.clips or [c for c, _ in hs]
    t0 = time.time()
    jobs = [(c, os.path.join(d, c + '.mp4'), a.sync_box, a.sync_interp) for c in ids]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(max(1, min(a.workers, 12, len(jobs)))) as pool:
        res = pool.map(eval_clip, jobs, chunksize=1)
    for r in res: r['set'] = setof.get(r['clip'])
    clips = {r['clip']: r for r in res}
    t_metrics = time.time() - t0
    failed = {c: r['error'] for c, r in clips.items() if 'error' in r}
    print(f'[{a.run}] metrics {t_metrics:.0f}s, failed: {failed}', flush=True)
    sys.path.insert(0, HERE); import judge
    jr = None
    if not a.no_judge:
        rp = probe(inputs(ids[0])['real'])
        items = [(c, os.path.join(d, c + '.mp4'), inputs(c)['real'], inputs(c)['lm']) for c in ids if 'error' not in clips[c]]
        jr = judge.judge_run(a.run, items, workers=a.judge_workers, hands=not a.no_hands_judge, W=rp['w'], H=rp['h'])
        for c, r in jr.items():
            clips[c]['realism'] = {k: r.get(k) for k in ('realism', 'realism_real', 'fooled_rate', 'trials', 'error') if k in r}
            h = r.get('hands') or {}
            clips[c]['realism']['hands'] = {k: h.get(k) for k in ('hands_score', 'hands_score_real', 'trials', 'skipped', 'error') if k in h}
    L = list(clips.values())
    doc = dict(run=a.run, dir=os.path.abspath(d), made=time.strftime('%F %T'), sync_box=a.sync_box, sync_interp=a.sync_interp,
               lmk=lmk_path(), inputs=inputs(ids[0])['source'], secs_metrics=round(t_metrics, 1),
               secs_total=round(time.time() - t0, 1), failed=failed,
               mean=aggregate(L, jr))
    for s in SETS:
        sub = [c for c in L if c.get('set') == s]
        if sub: doc['mean_' + s] = aggregate(sub, jr)
    doc['clips'] = clips
    out = a.out or os.path.join(RESULTS, f'{a.run}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f'.{os.getpid()}.tmp'
    json.dump(doc, open(tmp, 'w'), indent=1, default=float); os.replace(tmp, out)
    print(f'wrote {out}\n' + json.dumps(doc['mean'], indent=1), flush=True)


if __name__ == '__main__':
    main()
