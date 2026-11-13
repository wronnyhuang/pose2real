"""Cut + audio + landmarks + QC for windows of work/plan.json. usage: proc.py <id>...   (resumable: skips clips with qc.json)
Audio: copied from PRIOR clips/<prior_id>/ for re-cropped windows (same source + start frame = same audio), else cut fresh
(audio.py, PRIOR's method: mp4 AAC ch0, 16 kHz, 8.000 s + 10.000 s pad)."""
import json, os, sys, shutil, subprocess, time, numpy as np, cv2
SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, __import__('os').path.expandvars('$WORK/common')); sys.path.insert(0, SRC)
import lmk, audio as A, align, cut as C
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
D = __import__('os').path.expandvars('$WORK/data')
P = __import__('os').path.expandvars('$WORK/data')
PLAN = {w['id']: w for w in json.load(open(f'{D}/work/plan.json'))}


def second_faces(path):
    fl = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=lmk.FACE_MODEL),
         running_mode=V.RunningMode.IMAGE, num_faces=3, min_face_detection_confidence=0.4, min_face_presence_confidence=0.4))
    cap = cv2.VideoCapture(path); n = n2 = 0; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % 10 == 0:
            r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))); n += 1
            n2 += (sum(1 for lm in r.face_landmarks[1:] if np.ptp([p.x for p in lm]) > 0.05) > 0) if len(r.face_landmarks) > 1 else 0
        i += 1
    fl.close(); return n2 / max(n, 1)


def hand_stats(d):
    ok = d['hands_ok']; h = d['hands']; pose = d['pose']; W, H = 512, 640
    sw = np.linalg.norm((pose[:, 11, :2] - pose[:, 12, :2]) * [W, H], axis=1)
    sw = float(np.median(sw[d['pose_ok']])) if d['pose_ok'].any() else 1.0
    # hand_motion: mean per-frame displacement of the hand centroid (21 pts) over consecutive frames where the hand is tracked,
    # / shoulder width, x 30 -> shoulder widths per second; averaged over both hands (frames where tracked)
    mv = []
    for s in (0, 1):
        c = h[:, s, :, :2].mean(1) * [W, H]
        both = ok[1:, s] & ok[:-1, s]
        mv += list(np.linalg.norm(c[1:] - c[:-1], axis=1)[both] / sw * 30)
    size = []
    for s in (0, 1):
        P_ = h[ok[:, s], s, :, :2] * [W, H]
        if len(P_): size += list(np.linalg.norm(P_[:, 9] - P_[:, 0], axis=1))
    return dict(hand_frac=round(float(ok.any(1).mean()), 4), hand2_frac=round(float(ok.all(1).mean()), 4),
                hand_left_frac=round(float(ok[:, 0].mean()), 4), hand_right_frac=round(float(ok[:, 1].mean()), 4),
                hand_motion=round(float(np.mean(mv)), 4) if mv else 0.0,
                hand_size_px=round(float(np.median(size)), 1) if size else 0.0, shoulder_px=round(sw, 1))


def run(cid):
    w = PLAN[cid]; cd = f'{D}/clips/{cid}'; os.makedirs(cd, exist_ok=True)
    if os.path.exists(f'{cd}/qc.json'): return
    t0 = time.time(); st = w['st']
    if not os.path.exists(f'{cd}/real.mp4'): C.cut(w['video'], st, w['crop'], f'{cd}/real.mp4')
    if w['prior_id']:
        pd = f"{P}/clips/{w['prior_id']}"; pm = json.load(open(f'{pd}/meta.json'))
        assert pm['src_start_frame'] == st and pm['src_video'] == w['video'], cid
        for f in ('audio.wav', 'audio_pad.wav'): shutil.copyfile(f'{pd}/{f}', f'{cd}/{f}')
        asrc = dict(pm.get('audio', {}), copied_from=f"pose2real/data/clips/{w['prior_id']}")
        p = np.frombuffer(open(f'{cd}/audio_pad.wav', 'rb').read()[44:], np.int16)
    else:
        a, p = A.cut(w['video'], st / 30, 'mp4', 0); A.write_wav(f'{cd}/audio.wav', a); A.write_wav(f'{cd}/audio_pad.wav', p)
        asrc = dict(src='mp4 AAC track ch0', sr=16000, shift_samples=0)
    t1 = time.time()
    d = lmk.extract(f'{cd}/real.mp4'); lmk.save(d, f'{cd}/lm.tmp.npz'); os.replace(f'{cd}/lm.tmp.npz', f'{cd}/lm.npz'); t2 = time.time()
    T = len(d['face_ok']); f = d['face'][d['face_ok']]
    inside = float(((f[:, :, 0].min(1) > 0) & (f[:, :, 1].min(1) > -0.02) & (f[:, :, 0].max(1) < 1) & (f[:, :, 1].max(1) < 1)).mean()) if len(f) else 0.0
    fw = float(np.median(f[:, :, 0].max(1) - f[:, :, 0].min(1))) * 512 if len(f) else 0.0
    po = d['pose'][d['pose_ok']]
    sh_in = float(((po[:, [11, 12], 0] > 0) & (po[:, [11, 12], 0] < 1) & (po[:, [11, 12], 1] < 1)).all(1).mean()) if len(po) else 0.0
    m = align.mouth(d); pad = p.astype(np.float32) / 32767
    lags = {L: align.corr(m, align.env(pad, A.SR + int(round(L * A.SR / 30)))) for L in range(-6, 7)}
    best = max(lags, key=lambda k: -9 if np.isnan(lags[k]) else lags[k])
    hs = hand_stats(d)
    q = dict(frames=T, face_ok=round(float(d['face_ok'].mean()), 4), pose_ok=round(float(d['pose_ok'].mean()), 4),
             face_inside=round(inside, 4), face_width_px=round(fw, 1), shoulders_inside=round(sh_in, 4),
             second_face=round(second_faces(f'{cd}/real.mp4'), 3),
             av_corr_lag0=round(float(lags[0]), 3), av_best_lag=int(best), av_best_corr=round(float(lags[best]), 3),
             t_cut=round(t1 - t0, 1), t_lmk=round(t2 - t1, 1), **hs)
    why = []
    if T != 240: why.append(f'frames {T}')
    if q['face_ok'] < 0.95: why.append(f"face_ok {q['face_ok']:.2f}")
    if inside < 0.95: why.append(f'face leaves crop {inside:.2f}')
    if q['second_face'] > 0.3: why.append(f"second face {q['second_face']:.2f}")   # 0.1 flagged a sweater false positive (X0092)
    if q['pose_ok'] < 0.9: why.append(f"pose_ok {q['pose_ok']:.2f}")
    if sh_in < 0.9: why.append(f'shoulders leave crop {sh_in:.2f}')
    q['drop'] = '; '.join(why)
    c = w['crop']
    meta = dict(id=cid, identity=w['identity'], split=w['split'], set=w['set'], origin=w['origin'], prior_id=w['prior_id'],
                replaces=w.get('replaces'), file_id=w['file_id'], src_video=w['video'], src_start_s=round(st / 30, 4), src_start_frame=st,
                frames=240, fps=30, width=512, height=640,
                crop=dict(x=c['x'], y=c['y'], w=c['w'], h=c['h'], scale=c['scale'], src_wh=[1080, 1920]),
                speech_frac=w['speech'], frontal_frac=w['frontal'], kp_wrist_raised=w['raised'], kp_wrist_speed=w['speed'],
                face_width_px=q['face_width_px'], **hs, audio=asrc, qc=q)
    json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    json.dump(q, open(f'{cd}/qc.tmp', 'w')); os.replace(f'{cd}/qc.tmp', f'{cd}/qc.json')
    print(cid, 'ok' if not q['drop'] else 'DROP ' + q['drop'], 'hand', hs['hand_frac'], 'lmk', q['t_lmk'], flush=True)


if __name__ == '__main__':
    for cid in sys.argv[1:]:
        try: run(cid)
        except Exception as e:
            print('ERR', cid, repr(e), flush=True); open(f'{D}/work/errors.txt', 'a').write(f'{cid} {e!r}\n')
