"""Landmark / skeleton / hand fidelity of a generated video vs its conditioning lm.npz (both from ROOT/common/lmk.py).

All distances are computed in PIXELS of the actual frame: x*W, y*H (512 x 640 here — the frame is not square, so
normalised coordinates are never compared directly).

face_nme        mean over frames (both faces found) of mean L2 over the 468 face-mesh points (px)
                / conditioning inter-ocular distance (outer eye corners 33-263, same frame) x100
mouth_nme       same over the 40 FACEMESH_LIPS points
*_nme_pen       over ALL conditioning-face frames; a frame where the generated face is missing counts 100, errors clipped at 100
mouth_open_corr Pearson r over time of inner-lip opening (13-14) / face height (10-152), frames with both faces
det_rate        fraction of the conditioning-face frames in which the generated video has a face
pose_err        per frame: mean L2 (px) over pose points 0..16 with conditioning visibility > 0.5 and inside the frame,
                / conditioning shoulder width (clip median |11-12| px over conditioning pose frames) x100; mean over frames
                where both have a pose
pose_err_pen    missing generated pose = 100, frame errors clipped at 100
shoulder_err / elbow_err / wrist_err / arm_err
                exactly pose_err restricted to points {11,12} / {13,14} / {15,16} / {13,14,15,16}
hands (per slot 0 = anatomical left, 1 = right; a "hand instance" = one slot in one frame):
  n_hand_frames    hand instances in the conditioning
  hand_det_rate    P(generated has that hand | conditioning has it)
  hand_extra_rate  P(generated has that hand | conditioning does not)    (denominator n_nohand_frames)
  hand_err         mean L2 over the 21 points (px) / shoulder width x100, instances where both have the hand
  hand_err_hs      same / conditioning hand size (|wrist 0 - middle MCP 9| px, same frame) x100
  hand_err_pen     hand_err over all conditioning instances, missed hand = 100, clipped at 100 (hand_err_hs_pen alike)
  finger_pose_err  similarity-Procrustes (translation, uniform scale, rotation; no reflection) of the generated 21 points
                   onto the conditioning 21 points, mean residual L2 / conditioning hand size x100
  finger_angle_err_deg  mean |signed 2D bend angle difference| (wrapped to [0,180]) at 15 finger joints:
                   thumb CMC/MCP/IP (1,2,3), index/middle/ring/pinky MCP/PIP/DIP (5,6,7 / 9,10,11 / 13,14,15 / 17,18,19)
"""
import numpy as np

LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409,
        78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
POSE_GROUPS = dict(pose_err=list(range(17)), shoulder_err=[11, 12], elbow_err=[13, 14], wrist_err=[15, 16],
                   arm_err=[13, 14, 15, 16])
PEN = 100.0
# (prev, joint, next) for the 15 finger joints
JOINTS = [(0, 1, 2), (1, 2, 3), (2, 3, 4)] + [(p, j, j + 1) for b in (5, 9, 13, 17) for p, j in ((0, b), (b, b + 1), (b + 1, b + 2))]
HAND_SUM_KEYS = ('hand_det_rate', 'hand_extra_rate', 'hand_err', 'hand_err_hs', 'hand_err_pen', 'hand_err_hs_pen',
                 'finger_pose_err', 'finger_angle_err_deg')


def _pearson(a, b):
    if len(a) < 30 or np.std(a) < 1e-9 or np.std(b) < 1e-9: return None
    return float(np.corrcoef(a, b)[0, 1])


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def procrustes_residual(X, Y):
    """X: (N,21,2) generated, Y: (N,21,2) conditioning. Best similarity transform of X onto Y (no reflection);
    returns (N,) mean residual L2 in Y's units."""
    mx = X.mean(1, keepdims=True); my = Y.mean(1, keepdims=True)
    A = X - mx; B = Y - my
    out = np.zeros(len(X))
    for i in range(len(X)):
        a, b = A[i], B[i]
        na = (a ** 2).sum()
        if na < 1e-12: out[i] = np.linalg.norm(b, axis=1).mean(); continue
        M = a.T @ b                                   # 2x2
        U, S, Vt = np.linalg.svd(M)
        d = np.sign(np.linalg.det(U @ Vt)) or 1.0
        D = np.diag([1.0, d])
        R = U @ D @ Vt                                # a @ R ~ b
        s = (S * np.diag(D)).sum() / na
        out[i] = np.linalg.norm(s * a @ R - b, axis=1).mean()
    return out


def joint_angles(P):
    """P: (N,21,2) -> (N,15) signed bend angle in degrees at each finger joint."""
    pr = np.array([j[0] for j in JOINTS]); jo = np.array([j[1] for j in JOINTS]); nx = np.array([j[2] for j in JOINTS])
    u = P[:, jo] - P[:, pr]; v = P[:, nx] - P[:, jo]
    cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]; dot = (u * v).sum(-1)
    return np.degrees(np.arctan2(cross, dot))


def hand_compare(cond, gen, T, sc, sw):
    """per-clip hand metrics + counts (for instance-weighted run means)."""
    r = {}
    if 'hands' not in cond or 'hands' not in gen:
        r['hands_available'] = False; return r
    r['hands_available'] = True
    ch = cond['hands'][:T, :, :, :2] * sc; gh = gen['hands'][:T, :, :, :2] * sc
    cok = cond['hands_ok'][:T].astype(bool); gok = gen['hands_ok'][:T].astype(bool)
    both = cok & gok
    n_c = int(cok.sum()); n_none = int((~cok).sum()); n_b = int(both.sum())
    r.update(n_hand_frames=n_c, n_nohand_frames=n_none, n_hand_both=n_b, n_hand_extra=int((~cok & gok).sum()),
             n_hand_frames_slot=[int(cok[:, 0].sum()), int(cok[:, 1].sum())])
    r['hand_det_rate'] = _r(both.sum() / n_c) if n_c else None
    r['hand_extra_rate'] = _r((~cok & gok).sum() / n_none) if n_none else None
    hs = np.linalg.norm(ch[:, :, 0] - ch[:, :, 9], axis=-1) + 1e-6           # (T,2)
    d = np.linalg.norm(gh - ch, axis=-1).mean(-1)                           # (T,2) mean L2 over 21 pts, px
    e_sw = d / (sw + 1e-6) * 100; e_hs = d / hs * 100
    for k, e in (('hand_err', e_sw), ('hand_err_hs', e_hs)):
        r[k] = _r(e[both].mean()) if n_b else None
        r[k + '_pen'] = _r(np.where(gok, np.minimum(e, PEN), PEN)[cok].mean()) if n_c else None
    if n_b:
        X = gh[both]; Y = ch[both]
        r['finger_pose_err'] = _r((procrustes_residual(X, Y) / hs[both] * 100).mean())
        da = np.abs((joint_angles(X) - joint_angles(Y) + 180) % 360 - 180)
        r['finger_angle_err_deg'] = _r(da.mean(), 3)
        r['hand_size_px'] = _r(np.median(hs[both]), 2)
    else:
        r['finger_pose_err'] = None; r['finger_angle_err_deg'] = None
    return r


def compare(cond, gen, W=512, H=640):
    T = min(len(cond['face_ok']), len(gen['face_ok']))
    sc = np.array([W, H], np.float32)
    cf = cond['face'][:T, :, :2] * sc; gf = gen['face'][:T, :, :2] * sc
    cok = cond['face_ok'][:T].astype(bool); gok = gen['face_ok'][:T].astype(bool)
    both = cok & gok
    iod = np.linalg.norm(cf[:, 33] - cf[:, 263], axis=1) + 1e-6
    e_face = np.linalg.norm(gf[:, :468] - cf[:, :468], axis=2).mean(1) / iod * 100
    e_mouth = np.linalg.norm(gf[:, LIPS] - cf[:, LIPS], axis=2).mean(1) / iod * 100
    r = dict(frames=int(T), frames_gen=int(len(gen['face_ok'])), cond_face_frames=int(cok.sum()))
    r['det_rate'] = _r(gok[cok].mean()) if cok.any() else None
    for name, e in (('face_nme', e_face), ('mouth_nme', e_mouth)):
        r[name] = _r(e[both].mean()) if both.any() else None
        r[name + '_pen'] = _r(np.where(gok, np.minimum(e, PEN), PEN)[cok].mean()) if cok.any() else None
    r['iod_px'] = _r(np.median(iod[cok]), 2) if cok.any() else None

    def opening(f):
        return np.linalg.norm(f[:, 13] - f[:, 14], axis=1) / (np.linalg.norm(f[:, 10] - f[:, 152], axis=1) + 1e-6)
    c = _pearson(opening(gf)[both], opening(cf)[both]); r['mouth_open_corr'] = _r(c)

    # pose + arm groups
    cp = cond['pose'][:T]; gp = gen['pose'][:T]
    cpok = cond['pose_ok'][:T].astype(bool); gpok = gen['pose_ok'][:T].astype(bool)
    sw_all = np.linalg.norm(cp[:, 11, :2] * sc - cp[:, 12, :2] * sc, axis=1)
    sw = float(np.median(sw_all[cpok])) if cpok.any() else np.nan
    r['shoulder_width_px'] = _r(sw, 2)
    d_all = np.linalg.norm(gp[:, :, :2] * sc - cp[:, :, :2] * sc, axis=2)    # (T,33)
    valid = (cp[:, :, 3] > 0.5) & (cp[:, :, 0] >= 0) & (cp[:, :, 0] <= 1) & (cp[:, :, 1] >= 0) & (cp[:, :, 1] <= 1) & cpok[:, None]
    for name, J in POSE_GROUPS.items():
        m = valid[:, J]; has = m.any(1)
        e = np.where(has, (d_all[:, J] * m).sum(1) / np.maximum(m.sum(1), 1), np.nan) / (sw + 1e-6) * 100
        pb = has & gpok
        r[name] = _r(e[pb].mean()) if pb.any() else None
        if name == 'pose_err':
            r['pose_points_mean'] = _r(m.sum(1)[has].mean(), 2) if has.any() else 0
            r['pose_err_pen'] = _r(np.where(gpok, np.minimum(np.nan_to_num(e, nan=PEN), PEN), PEN)[has].mean()) if has.any() else None
            r['pose_det_rate'] = _r(gpok[has].mean()) if has.any() else None
        else:
            r[name + '_frames'] = int(pb.sum())
    r.update(hand_compare(cond, gen, T, sc, sw))
    return r


def hand_run_mean(clips):
    """Instance-weighted run means over clip dicts (landmarks sub-dicts). Clips without hands contribute nothing.
    det_rate / *_pen weighted by n_hand_frames, extra_rate by n_nohand_frames, the both-present errors by n_hand_both
    (= the pooled mean over all measured hand instances)."""
    out = {}
    L = [c for c in clips if c.get('hands_available')]
    if not L: return out
    def wmean(k, wk):
        v = [(c[k], c[wk]) for c in L if c.get(k) is not None and c.get(wk)]
        if not v: return None
        return _r(sum(a * w for a, w in v) / sum(w for _, w in v))
    for k in ('hand_det_rate', 'hand_err_pen', 'hand_err_hs_pen'): out[k] = wmean(k, 'n_hand_frames')
    out['hand_extra_rate'] = wmean('hand_extra_rate', 'n_nohand_frames')
    for k in ('hand_err', 'hand_err_hs', 'finger_pose_err', 'finger_angle_err_deg'): out[k] = wmean(k, 'n_hand_both')
    out['n_hand_frames'] = int(sum(c.get('n_hand_frames', 0) for c in L))
    out['n_hand_both'] = int(sum(c.get('n_hand_both', 0) for c in L))
    out['n_hand_clips'] = int(sum(1 for c in L if c.get('n_hand_frames')))
    return out
