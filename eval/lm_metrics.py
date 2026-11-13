"""Landmark / skeleton fidelity of a generated video vs its conditioning lm.npz (both from common/lmk.py).

face_nme        mean over frames (both faces found) of mean L2 error over the 468 face-mesh points (x,y px)
                / conditioning inter-ocular distance (outer eye corners 33-263, same frame) x100
mouth_nme       same over the 40 FACEMESH_LIPS points (outer + inner lip)
*_nme_pen       same but over ALL frames with a conditioning face; a frame where the generated face is missing counts
                as 100 (= one inter-ocular distance); every frame error is clipped at 100
mouth_open_corr Pearson r over time of inner-lip opening (13-14) / face height (10-152), frames with both faces
pose_err        mean L2 error (px) over pose points 0..16 (nose, eyes, ears, mouth, shoulders, elbows, wrists) with
                conditioning visibility > 0.5 and inside the frame, / conditioning shoulder width (clip median 11-12) x100
pose_err_pen    missing generated pose = 100, frame errors clipped at 100
det_rate        fraction of the conditioning-face frames in which the generated video has a face
"""
import numpy as np

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
POSE_J = list(range(17))
PEN = 100.0


def _pearson(a, b):
    if len(a) < 30 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
    return float(np.corrcoef(a, b)[0, 1])


def compare(cond, gen, W=384, H=384):
    T = min(len(cond['face_ok']), len(gen['face_ok']))
    sc = np.array([W, H], np.float32)
    cf = cond['face'][:T, :, :2] * sc; gf = gen['face'][:T, :, :2] * sc
    cok = cond['face_ok'][:T].astype(bool); gok = gen['face_ok'][:T].astype(bool)
    both = cok & gok
    iod = np.linalg.norm(cf[:, 33] - cf[:, 263], axis=1) + 1e-6
    e_face = np.linalg.norm(gf[:, :468] - cf[:, :468], axis=2).mean(1) / iod * 100
    e_mouth = np.linalg.norm(gf[:, LIPS] - cf[:, LIPS], axis=2).mean(1) / iod * 100
    r = dict(frames=int(T), cond_face_frames=int(cok.sum()))
    r['det_rate'] = round(float(gok[cok].mean()), 4) if cok.any() else None
    for name, e in (('face_nme', e_face), ('mouth_nme', e_mouth)):
        r[name] = round(float(e[both].mean()), 4) if both.any() else None
        pen = np.where(gok, np.minimum(e, PEN), PEN)[cok]
        r[name + '_pen'] = round(float(pen.mean()), 4) if cok.any() else None

    def opening(f):
        return np.linalg.norm(f[:, 13] - f[:, 14], axis=1) / (np.linalg.norm(f[:, 10] - f[:, 152], axis=1) + 1e-6)
    oc, og = opening(cf), opening(gf)
    c = _pearson(og[both], oc[both]); r['mouth_open_corr'] = round(c, 4) if c is not None else None

    # pose
    cp = cond['pose'][:T]; gp = gen['pose'][:T]
    cpok = cond['pose_ok'][:T].astype(bool); gpok = gen['pose_ok'][:T].astype(bool)
    cxy = cp[:, POSE_J, :2] * sc; gxy = gp[:, POSE_J, :2] * sc
    sw_all = np.linalg.norm(cp[:, 11, :2] * sc - cp[:, 12, :2] * sc, axis=1)
    sw = float(np.median(sw_all[cpok])) if cpok.any() else np.nan
    m = (cp[:, POSE_J, 3] > 0.5) & (cp[:, POSE_J, 0] >= 0) & (cp[:, POSE_J, 0] <= 1) & (cp[:, POSE_J, 1] >= 0) & (cp[:, POSE_J, 1] <= 1)
    m &= cpok[:, None]
    has = m.any(1)
    d = np.linalg.norm(gxy - cxy, axis=2)
    e_pose = np.where(has, (d * m).sum(1) / np.maximum(m.sum(1), 1), np.nan) / (sw + 1e-6) * 100
    pb = has & gpok
    r['shoulder_width_px'] = round(sw, 2) if np.isfinite(sw) else None
    r['pose_points_mean'] = round(float(m.sum(1)[has].mean()), 2) if has.any() else 0
    r['pose_err'] = round(float(e_pose[pb].mean()), 4) if pb.any() else None
    r['pose_err_pen'] = round(float(np.where(gpok, np.minimum(np.nan_to_num(e_pose, nan=PEN), PEN), PEN)[has].mean()), 4) if has.any() else None
    r['pose_det_rate'] = round(float(gpok[has].mean()), 4) if has.any() else None
    return r
