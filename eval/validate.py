#!/usr/bin/env python3
"""SyncNet harness validation on the 12 real heldout clips -> eval/results_validation.json
  real + own audio           : LSE-C high, offset ~0
  audio delayed / advanced 5 frames (25 fps, 200 ms; cut from audio_pad.wav so no zeros): offset moves -5 / +5
  real + another clip's audio (different identity): LSE-C low
  same, with the original S3FD face box instead of the MediaPipe box (crop-policy check)
"""
import json, os, sys, time
from multiprocessing import get_context
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def job(args):
    cid, other = args
    import run_eval, sync
    inp = run_eval.inputs(cid); oinp = run_eval.inputs(other)
    if inp['lm']:
        z = np.load(inp['lm']); cond = {k: z[k] for k in z.files}
    else:
        cond, _ = run_eval.cached_lmk(inp['real'])
    fr, fps = sync.read_frames(inp['real'])
    boxes = sync.mp_boxes(cond['face'], cond['face_ok'].astype(bool), 384, 384)
    _, a = sync.read_wav(inp['audio']); _, ap = sync.read_wav(inp['audio_pad']); _, ao = sync.read_wav(oinp['audio'])
    r = dict(clip=cid, other=other)
    r['own'] = sync.sync_for(fr, fps, boxes, a)
    # pad starts 1.0 s before frame 0 (16000 samples). audio delayed by k frames = window starts k*640 samples earlier
    for k in (5, -5):
        s = 16000 - k * 640
        r[f'audio_delayed_{k:+d}'] = sync.sync_for(fr, fps, boxes, ap[s:s + len(a)])
    r['mismatched'] = sync.sync_for(fr, fps, boxes, ao[:len(a)])
    idx = sync.resample_idx(len(fr), fps); b25 = sync.s3fd_boxes(fr[idx])
    r['own_s3fd'] = sync.score(sync.crops(fr[idx], b25), a); r['own_s3fd']['box_det_rate'] = float(np.mean(~np.isnan(b25[:, 0])))
    bm = boxes[idx]; ok = ~np.isnan(b25[:, 0]) & ~np.isnan(bm[:, 0])
    sz = lambda b: np.maximum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])
    r['box_size_ratio_mp_over_s3fd'] = float(np.median(sz(bm[ok]) / sz(b25[ok]))) if ok.any() else None
    r['inputs'] = inp['source']
    return r


def main():
    import run_eval
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    try: os.nice(10)
    except Exception: pass
    ids = run_eval.heldout_ids()
    ident = lambda c: c.split('_')[1]
    pairs = []
    for i, c in enumerate(ids):
        j = (i + 1) % len(ids)
        while ident(ids[j]) == ident(c): j = (j + 1) % len(ids)
        pairs.append((c, ids[j]))
    t0 = time.time()
    with get_context('spawn').Pool(12) as p: res = p.map(job, pairs, chunksize=1)
    keys = ['own', 'audio_delayed_+5', 'audio_delayed_-5', 'mismatched', 'own_s3fd']
    summ = {}
    for k in keys:
        summ[k] = dict(lse_c=round(float(np.mean([r[k]['lse_c'] for r in res])), 3),
                       lse_d=round(float(np.mean([r[k]['lse_d'] for r in res])), 3),
                       offsets=[r[k]['offset_frames'] for r in res])
    summ['box_size_ratio_mp_over_s3fd'] = round(float(np.median([r['box_size_ratio_mp_over_s3fd'] for r in res])), 3)
    doc = dict(made=time.strftime('%F %T'), secs=round(time.time() - t0, 1), summary=summ, clips=res)
    out = os.path.join(HERE, 'results_validation.json'); json.dump(doc, open(out + '.tmp', 'w'), indent=1); os.replace(out + '.tmp', out)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
