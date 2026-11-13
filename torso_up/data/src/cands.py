"""Score every 240-frame window (step 15) of every Seamless dev video for gesture selection -> work/cands.json.
Per window: speech (VAD), frontal (pose2real rule), raised (share of frames with a wrist above hip - 0.3*torso, keypoint conf > .5),
speed (median over frames of the faster wrist's 5-frame displacement / shoulder width / 5), handvis (share of frames with
mean hand-keypoint conf > .5 for >= 1 hand), body (valid box + shoulders), plus the torso crop (framing.crop) and face_px
(ear-to-ear * scale)."""
import json, glob, os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import framing
SEAM = __import__('os').path.expandvars('$SEAMLESS_ROOT')
D = __import__('os').path.expandvars('$WORK/data')
L = 240
def one(vid):
    out = []
    fid = os.path.basename(vid)[:-4]; base = os.path.dirname(os.path.dirname(vid)); pid = fid.split('_')[-1]
    try:
        kp = np.load(f'{base}/boxes_and_keypoints/keypoints/{fid}.npy'); vb = np.load(f'{base}/boxes_and_keypoints/is_valid_box/{fid}.npy').reshape(-1)
    except Exception as e:
        print('skip', fid, e); return []
    vp = f'{base}/metadata/vad/{fid}.jsonl'
    vad = [json.loads(l) for l in open(vp)] if os.path.exists(vp) else []
    T = len(kp); sp = np.zeros(T)
    for s in vad: sp[int(s['start'] * 30):int(s['end'] * 30)] = 1
    nose, le, re_ = kp[:, 0], kp[:, 3], kp[:, 4]
    ed = np.abs(le[:, 0] - re_[:, 0]); mid = (le[:, 0] + re_[:, 0]) / 2
    frontal = (le[:, 2] > 0.5) & (re_[:, 2] > 0.5) & (nose[:, 2] > 0.6) & (np.abs(nose[:, 0] - mid) / np.maximum(ed, 1) < 0.18) & (vb[:T] > 0)
    sw = np.abs(kp[:, 5, 0] - kp[:, 6, 0]) + 1e-3
    shy = (kp[:, 5, 1] + kp[:, 6, 1]) / 2; hipy = (kp[:, 11, 1] + kp[:, 12, 1]) / 2; torso = np.maximum(hipy - shy, 1)
    wr = kp[:, [9, 10]]
    raised = ((wr[:, :, 1] < (hipy - 0.3 * torso)[:, None]) & (wr[:, :, 2] > 0.5)).any(1)
    disp = np.zeros((T, 2)); disp[5:] = np.linalg.norm(wr[5:, :, :2] - wr[:-5, :, :2], axis=2) / sw[5:, None] / 5
    disp[(wr[:, :, 2] < 0.5)] = 0
    hc = np.stack([kp[:, 91:112, 2].mean(1), kp[:, 112:133, 2].mean(1)], 1)
    handvis = (hc > 0.5).any(1)
    body = (vb[:T] > 0) & (kp[:, 5, 2] > 0.5) & (kp[:, 6, 2] > 0.5)
    for st in range(30, T - L - 30, 15):
        w = slice(st, st + L)
        if body[w].mean() < 0.95: continue
        cr = framing.crop(kp[w], vb[w])
        if cr is None: continue
        fm = frontal[w]
        e = float(np.median(ed[w][fm])) if fm.any() else 0.0
        out.append(dict(file_id=fid, video=vid, identity=pid, st=st, speech=round(float(sp[w].mean()), 3), frontal=round(float(fm.mean()), 3),
                        raised=round(float(raised[w].mean()), 3), speed=round(float(np.median(disp[w].max(1))), 4),
                        handvis=round(float(handvis[w].mean()), 3), face_px=round(e * cr['scale'], 1), crop=cr))
    return out
if __name__ == '__main__':
    from multiprocessing import Pool
    with Pool(16) as p: res = p.map(one, sorted(glob.glob(f'{SEAM}/*/dev/video/*.mp4')))
    out = [w for r in res for w in r]
    json.dump(out, open(f'{D}/work/cands.json', 'w'))
    print('windows', len(out))
