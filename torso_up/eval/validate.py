#!/usr/bin/env python3
"""SyncNet harness validation on the 20 real heldout clips (torso-up 512x640) -> eval/results_validation.json
For each face-crop resize mode (linear = pose2real / reference, lanczos, cubic):
  real + own audio                      : LSE-C high, offset ~0
  audio delayed / advanced 5 frames     : offset moves -5 / +5   (cut from audio_pad.wav, so no zeros)
  real + another identity's audio       : LSE-C low
plus (--s3fd) the original S3FD box (detection at scale 0.5) with the linear resize, the crop size in px, and the bust
(pose2real, 384x384) own-audio LSE-C of the same12 clips for a clip-by-clip comparison.
    python eval/validate.py [--interps linear lanczos cubic] [--s3fd] [--workers 12]   (per (clip,case) units cached in cache/validate/)
"""
import argparse, json, os, sys, time
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
PRIOR_VAL = __import__('os').path.expandvars('$WORK/eval/results_validation.json')


VCACHE = os.path.join(HERE, 'cache', 'validate')


def job(args):
    """one (clip, case) unit; case = an interp name (own/+-5/mismatched with that resize) or 's3fd'. Cached per unit."""
    cid, other, case = args
    import cv2; cv2.setNumThreads(1)
    import run_eval, sync
    inp = run_eval.inputs(cid); oinp = run_eval.inputs(other)
    import hashlib
    key = hashlib.md5(f'v1|{run_eval.sig(inp["real"])}|{run_eval.sig(inp["lm"])}|{run_eval.sig(oinp["audio"])}|{case}'.encode()).hexdigest()
    cp = os.path.join(VCACHE, f'{cid}_{case}_{key[:10]}.json')
    if os.path.exists(cp): return json.load(open(cp))
    cond = run_eval.load_cond(inp)
    fr, fps = sync.read_frames(inp['real'])
    hh, ww = fr.shape[1:3]
    boxes = sync.mp_boxes(cond['face'], cond['face_ok'].astype(bool), ww, hh)
    _, a = sync.read_wav(inp['audio']); _, ap = sync.read_wav(inp['audio_pad']); _, ao = sync.read_wav(oinp['audio'])
    r = dict(clip=cid, other=other, case=case)
    if case != 's3fd':
        ip = case
        r[f'own_{ip}'] = sync.sync_for(fr, fps, boxes, a, interp=ip)
        # pad starts 1.0 s before frame 0 (16000 samples). audio delayed by k frames (25 fps) = window starts k*640 samples earlier
        for k in (5, -5):
            s = 16000 - k * 640
            r[f'delayed_{k:+d}_{ip}'] = sync.sync_for(fr, fps, boxes, ap[s:s + len(a)], interp=ip)
        r[f'mismatched_{ip}'] = sync.sync_for(fr, fps, boxes, ao[:len(a)], interp=ip)
        r['crop_px'] = sync.crop_size_px(boxes)
        r['face_w_px'] = float(np.median((boxes[:, 2] - boxes[:, 0])[~np.isnan(boxes[:, 0])] / sync.MP_BOX_SCALE))
    else:
        idx = sync.resample_idx(len(fr), fps); b25 = sync.s3fd_boxes(fr[idx], scale=S3FD_SCALE)
        c = sync.crops(fr[idx], b25, 'linear')
        r['own_s3fd'] = sync.score(c, a) if c is not None else dict(error='no s3fd box')
        r['own_s3fd']['box_det_rate'] = float(np.mean(~np.isnan(b25[:, 0])))
        bm = boxes[idx]; ok = ~np.isnan(b25[:, 0]) & ~np.isnan(bm[:, 0])
        sz = lambda b: np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
        r['box_size_ratio_mp_over_s3fd'] = float(np.median(sz(bm[ok]) / sz(b25[ok]))) if ok.any() else None
    os.makedirs(VCACHE, exist_ok=True)
    tmp = cp + f'.{os.getpid()}.tmp'; json.dump(r, open(tmp, 'w')); os.replace(tmp, cp)
    return r


S3FD_SCALE = 0.5


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--interps', nargs='*', default=['linear', 'lanczos', 'cubic'])
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--s3fd', action='store_true', help='also the S3FD box (scale 0.5; slow on CPU)'); a = ap.parse_args()
    import run_eval
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    try: os.nice(10)
    except Exception: pass
    hs = run_eval.heldout(); ids = [c for c, _ in hs]; setof = dict(hs)
    ident = lambda c: c.split('_')[1]
    pairs = []
    for i, c in enumerate(ids):
        j = (i + 1) % len(ids)
        while ident(ids[j]) == ident(c): j = (j + 1) % len(ids)
        pairs.append((c, ids[j]))
    cases = list(a.interps) + (['s3fd'] if a.s3fd else [])
    t0 = time.time()
    with get_context('spawn').Pool(min(a.workers, 12)) as p:
        units = p.map(job, [(c, o, k) for k in cases for c, o in pairs], chunksize=1)
    res = []
    for c, o in pairs:
        r = dict(clip=c, other=o, inputs=run_eval.inputs(c)['source'], box_size_ratio_mp_over_s3fd=None)
        for u in units:
            if u['clip'] == c: r.update({k: v for k, v in u.items() if k not in ('case',)})
        res.append(r)
    try:
        pv = {c['clip']: c['own']['lse_c'] for c in json.load(open(PRIOR_VAL))['clips']}
    except Exception: pv = {}
    for r in res:
        r['set'] = setof[r['clip']]; r['bust_own_lse_c'] = pv.get(r['clip'])

    def summ(rows):
        out = {}
        keys = [k for k in rows[0] if isinstance(rows[0][k], dict) and 'lse_c' in rows[0][k]]
        for k in keys:
            ok = [r[k] for r in rows if 'lse_c' in r[k]]
            out[k] = dict(lse_c=round(float(np.mean([x['lse_c'] for x in ok])), 3),
                          lse_d=round(float(np.mean([x['lse_d'] for x in ok])), 3), offsets=[x['offset_frames'] for x in ok])
        for ip in a.interps:
            own = np.array([r[f'own_{ip}']['offset_frames'] for r in rows])
            out[f'shift_exact_{ip}'] = int(sum((np.array([r[f'delayed_+5_{ip}']['offset_frames'] for r in rows]) == own - 5).tolist()
                                              + (np.array([r[f'delayed_-5_{ip}']['offset_frames'] for r in rows]) == own + 5).tolist()))
            out[f'separates_{ip}'] = int(sum(r[f'own_{ip}']['lse_c'] > r[f'mismatched_{ip}']['lse_c'] + 1.0 for r in rows))
        b = [(r['bust_own_lse_c'], r[f'own_{a.interps[0]}']['lse_c']) for r in rows if r['bust_own_lse_c'] is not None]
        if b: out['bust_vs_torso_same_clips'] = dict(n=len(b), bust=round(float(np.mean([x for x, _ in b])), 3),
                                                     **{f'torso_{ip}': round(float(np.mean([r[f'own_{ip}']['lse_c'] for r in rows if r['bust_own_lse_c'] is not None])), 3) for ip in a.interps})
        out['crop_px_median'] = round(float(np.median([r['crop_px'] for r in rows])), 1)
        out['face_w_px_median'] = round(float(np.median([r['face_w_px'] for r in rows])), 1)
        rr = [r['box_size_ratio_mp_over_s3fd'] for r in rows if r.get('box_size_ratio_mp_over_s3fd')]
        out['box_size_ratio_mp_over_s3fd'] = round(float(np.median(rr)), 3) if rr else None
        out['n'] = len(rows)
        return out
    doc = dict(made=time.strftime('%F %T'), secs=round(time.time() - t0, 1), summary=summ(res),
               summary_same12=summ([r for r in res if r['set'] == 'same12']),
               summary_gesture8=summ([r for r in res if r['set'] == 'gesture8']) if any(r['set'] == 'gesture8' for r in res) else None,
               clips=res)
    out = os.path.join(HERE, 'results_validation.json'); json.dump(doc, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
    print(json.dumps({k: doc[k] for k in ('summary', 'summary_same12', 'summary_gesture8')}, indent=1))


if __name__ == '__main__':
    main()
