#!/usr/bin/env python3
"""pose2real eval harness (CPU only).
    python eval/run_eval.py --run <run> [--dir <folder with <clip_id>.mp4>] [--no-judge] [--workers N] [--sync-box mp|s3fd]
default dir = ROOT/runs/<run>/out ; writes eval/results/<run>.json (atomic).  See eval/README.md.
"""
import argparse, hashlib, json, os, subprocess, sys, tempfile, time, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
T4448 = __import__('os').path.expandvars('$WORK')
CACHE = os.path.join(HERE, 'cache')
RESULTS = os.path.join(HERE, 'results')
SCRATCH = os.path.join(HERE, 'scratch')
N_FRAMES, SIZE, FPS = 240, 384, 30


def heldout_ids():
    for m in (os.path.join(ROOT, 'data', 'manifest.json'), os.path.join(T4448, 'data', 'manifest.json')):
        try:
            ids = [c['id'] for c in json.load(open(m))['clips'] if c.get('split') == 'heldout']
            if len(ids) == 12: return sorted(ids)
        except Exception: pass
    raise SystemExit('no heldout manifest')


def seed_done(): return os.path.exists(os.path.join(ROOT, 'data', 'SEED_DONE'))


def inputs(cid):
    """real video, audio wav, padded audio wav, conditioning npz (or None = compute from real & cache)."""
    c = os.path.join(ROOT, 'data', 'clips', cid)
    if seed_done() and os.path.exists(os.path.join(c, 'lm.npz')):
        return dict(real=os.path.join(c, 'real.mp4'), audio=os.path.join(c, 'audio.wav'),
                    audio_pad=os.path.join(c, 'audio_pad.wav'), lm=os.path.join(c, 'lm.npz'), source='ROOT/data/clips')
    return dict(real=os.path.join(T4448, 'data', 'pairs', cid, 'real.mp4'), audio=os.path.join(SCRATCH, 'audio', cid + '.wav'),
                audio_pad=os.path.join(SCRATCH, 'audio', cid + '_pad.wav'), lm=None, source='dev fallback (render2real pairs + scratch audio)')


def sig(p):
    st = os.stat(p); return f'{os.path.realpath(p)}|{st.st_size}|{st.st_mtime_ns}'


def lmk_mod():
    import importlib.util
    p = os.path.join(ROOT, 'common', 'lmk.py')
    spec = importlib.util.spec_from_file_location('pose2real_lmk', p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m, sig(p)


def cached_lmk(video):
    m, lsig = lmk_mod()
    key = hashlib.md5(f'{sig(video)}|{lsig}'.encode()).hexdigest()
    p = os.path.join(CACHE, 'lmk', key + '.npz')
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}, True
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


def normalize(path, tmpdir):
    """ensure 384x384 / 30 fps / 240 frames; returns (path to use, note or None)."""
    p = probe(path)
    if p['w'] == SIZE and p['h'] == SIZE and abs(p['fps'] - FPS) < 0.01 and p['n'] == N_FRAMES: return path, None
    out = os.path.join(tmpdir, 'norm.mp4')
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', path, '-an', '-vf',
                    f'scale={SIZE}:{SIZE}:flags=area,fps={FPS},tpad=stop_mode=clone:stop=-1', '-frames:v', str(N_FRAMES),
                    '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', out], check=True)
    return out, f'normalized from {p}'


def eval_clip(job):
    cid, gen, box_mode = job
    import cv2
    cv2.setNumThreads(1)
    sys.path.insert(0, HERE)
    import sync, lm_metrics
    t0 = time.time(); res = dict(clip=cid)
    try:
        inp = inputs(cid); res['inputs'] = inp['source']
        if not os.path.exists(gen): return dict(clip=cid, error=f'missing output {gen}')
        with tempfile.TemporaryDirectory(prefix=f'ev_{cid}_', dir=os.environ.get('EVAL_TMP')) as td:
            v, note = normalize(gen, td)
            if note: res['normalized'] = note
            # ---- landmarks
            g, hit = cached_lmk(v) if v == gen else (lmk_mod()[0].extract(v), False)
            if inp['lm']:
                z = np.load(inp['lm']); cond = {k: z[k] for k in z.files}
            else:
                cond, _ = cached_lmk(inp['real'])
            res['landmarks'] = lm_metrics.compare(cond, g, SIZE, SIZE)
            res['lmk_cached'] = hit
            # ---- syncnet (cached per video+audio+box mode)
            skey = hashlib.md5(f'v2|{sig(gen)}|{sig(inp["audio"])}|{box_mode}'.encode()).hexdigest()
            sp = os.path.join(CACHE, 'sync', skey + '.json')
            if os.path.exists(sp):
                res['syncnet'] = json.load(open(sp)); res['sync_cached'] = True
            else:
                mkv, (sr, audio) = sync.mux_and_extract(v, inp['audio'], td)
                frames, fps = sync.read_frames(mkv)
                if box_mode == 's3fd':
                    idx = sync.resample_idx(len(frames), fps)
                    b25 = sync.s3fd_boxes(frames[idx]); c = sync.crops(frames[idx], b25)
                    s = sync.score(c, audio, sr) if c is not None else dict(error='no face box')
                    s['box_det_rate'] = round(float(np.mean(~np.isnan(b25[:, 0]))), 4)
                else:
                    boxes = sync.mp_boxes(g['face'], g['face_ok'].astype(bool), SIZE, SIZE)
                    s = sync.sync_for(frames, fps, boxes, audio, sr)
                s['box'] = box_mode
                os.makedirs(os.path.dirname(sp), exist_ok=True)
                if 'error' not in s:
                    json.dump(s, open(sp + '.tmp', 'w')); os.replace(sp + '.tmp', sp)
                res['syncnet'] = s; res['sync_cached'] = False
    except Exception as e:
        import traceback
        res['error'] = f'{type(e).__name__}: {e}'; res['trace'] = traceback.format_exc()[-1500:]
    res['secs'] = round(time.time() - t0, 1)
    return res


def mean_of(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return round(float(np.mean(v)), 4) if v else None


def aggregate(clips, judge_sum):
    ok = [c for c in clips.values() if 'error' not in c]
    sy = lambda k: mean_of([c['syncnet'].get(k) for c in ok if 'error' not in c.get('syncnet', {'error': 1})])
    lm = lambda k: mean_of([c['landmarks'].get(k) for c in ok])
    m = dict(n_clips=len(ok), n_failed=len(clips) - len(ok),
             syncnet=dict(lse_c=sy('lse_c'), lse_d=sy('lse_d'), offset_frames=sy('offset_frames'),
                          abs_offset_frames=mean_of([abs(c['syncnet']['offset_frames']) for c in ok if 'offset_frames' in c.get('syncnet', {})]),
                          lse_d_at0=sy('lse_d_at0')),
             landmarks={k: lm(k) for k in ('face_nme', 'mouth_nme', 'pose_err', 'mouth_open_corr', 'det_rate',
                                           'face_nme_pen', 'mouth_nme_pen', 'pose_err_pen', 'pose_det_rate')})
    m['realism'] = judge_sum if judge_sum else None
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True); ap.add_argument('--dir')
    ap.add_argument('--no-judge', action='store_true'); ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--judge-workers', type=int, default=4)
    ap.add_argument('--sync-box', default='mp', choices=['mp', 's3fd'])
    ap.add_argument('--clips', nargs='*'); ap.add_argument('--out')
    a = ap.parse_args()
    try: os.nice(10)
    except Exception: pass
    os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('SYNC_THREADS', '2')
    d = a.dir or os.path.join(ROOT, 'runs', a.run, 'out')
    ids = a.clips or heldout_ids()
    t0 = time.time()
    jobs = [(c, os.path.join(d, c + '.mp4'), a.sync_box) for c in ids]
    import multiprocessing as mp
    with mp.get_context('spawn').Pool(max(1, min(a.workers, 16, len(jobs)))) as pool:
        res = pool.map(eval_clip, jobs, chunksize=1)
    clips = {r['clip']: r for r in res}
    t_metrics = time.time() - t0
    print(f'[{a.run}] metrics {t_metrics:.0f}s, failed: {[c for c, r in clips.items() if "error" in r]}', flush=True)
    jsum = None
    if not a.no_judge:
        sys.path.insert(0, HERE); import judge
        items = [(c, os.path.join(d, c + '.mp4'), inputs(c)['real']) for c in ids if 'error' not in clips[c]]
        jr = judge.judge_run(a.run, items, workers=a.judge_workers)
        for c, r in jr.items():
            clips[c]['realism'] = {k: r.get(k) for k in ('realism', 'realism_real', 'fooled_rate', 'trials', 'error') if k in r}
        jsum = judge.summarize(jr); jsum['model'] = judge.MODEL
    doc = dict(run=a.run, dir=os.path.abspath(d), made=time.strftime('%F %T'), sync_box=a.sync_box,
               inputs=inputs(ids[0])['source'], secs_metrics=round(t_metrics, 1), secs_total=round(time.time() - t0, 1),
               mean=aggregate(clips, jsum), clips=clips)
    out = a.out or os.path.join(RESULTS, f'{a.run}.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + f'.{os.getpid()}.tmp'
    json.dump(doc, open(tmp, 'w'), indent=1, default=float); os.replace(tmp, out)
    print(f'wrote {out}\n' + json.dumps(doc['mean'], indent=1), flush=True)


if __name__ == '__main__':
    main()
