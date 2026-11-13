"""Cut + audio + landmarks + QC for one selected window (work/sel2.json). usage: proc.py <id>..."""
import json, os, sys, subprocess, time, numpy as np, cv2
sys.path.insert(0, __import__('os').path.expandvars('$WORK/common')); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lmk, audio as A, align
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
D = __import__('os').path.expandvars('$WORK/data')
SEL = {w['id']: w for w in json.load(open(f'{D}/work/sel2.json'))}

def second_faces(path):
    fl = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=lmk.FACE_MODEL),
         running_mode=V.RunningMode.IMAGE, num_faces=3, min_face_detection_confidence=0.4, min_face_presence_confidence=0.4))
    cap = cv2.VideoCapture(path); n = n2 = 0; i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i % 10 == 0:
            r = fl.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))); n += 1
            n2 += sum(1 for lm in r.face_landmarks[1:] if np.ptp([p.x for p in lm]) > 0.08) > 0 if len(r.face_landmarks) > 1 else 0
        i += 1
    fl.close(); return n2 / max(n, 1)

def run(cid):
    w = SEL[cid]; cd = f'{D}/clips/{cid}'; os.makedirs(cd, exist_ok=True)
    if os.path.exists(f'{cd}/qc.json'): return
    t0 = time.time(); S, x0, y0 = w['S'], w['x0'], w['y0']; st = w['st'] / 30
    vf = f'trim=start_frame=0:end_frame=240,setpts=PTS-STARTPTS,crop={S}:{S}:{x0}:{y0},scale=384:384:flags=lanczos'
    subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{st:.4f}', '-i', w['video'], '-vf', vf, '-an', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '16', '-pix_fmt', 'yuv420p', '-r', '30', '-y', f'{cd}/real.tmp.mp4'], check=True)
    os.replace(f'{cd}/real.tmp.mp4', f'{cd}/real.mp4')
    a, p = A.cut(w['video'], st, 'mp4', 0); A.write_wav(f'{cd}/audio.wav', a); A.write_wav(f'{cd}/audio_pad.wav', p)
    meta = dict(id=cid, identity=w['identity'], split=w['split'], origin='expand', file_id=w['file_id'], src_video=w['video'],
                src_start_s=round(st, 4), src_start_frame=w['st'], frames=240, fps=30, size=384,
                crop=dict(x=x0, y=y0, S=S, src_wh=[1080, 1920], shift_frac=w['shift']),
                speech_frac=w['speech'], frontal_frac=w['frontal'], strict_gates=w['strict'],
                audio=dict(src='mp4 AAC track ch0', sr=16000, shift_samples=0))
    t1 = time.time(); d = lmk.extract(f'{cd}/real.mp4'); lmk.save(d, f'{cd}/lm.tmp.npz'); os.replace(f'{cd}/lm.tmp.npz', f'{cd}/lm.npz'); t2 = time.time()
    T = len(d['face_ok']); f = d['face'][d['face_ok']]
    inside = float(((f[:, :, 0].min(1) > 0) & (f[:, :, 1].min(1) > -0.02) & (f[:, :, 0].max(1) < 1) & (f[:, :, 1].max(1) < 1)).mean()) if len(f) else 0.0
    fw = float(np.median(f[:, :, 0].max(1) - f[:, :, 0].min(1))) if len(f) else 0.0
    m = align.mouth(d); pad = np.frombuffer(p.tobytes(), np.int16).astype(np.float32) / 32767
    lags = {L: align.corr(m, align.env(pad, A.SR + int(round(L * A.SR / 30)))) for L in range(-6, 7)}
    best = max(lags, key=lambda k: -9 if np.isnan(lags[k]) else lags[k])
    q = dict(frames=T, face_ok=float(d['face_ok'].mean()), pose_ok=float(d['pose_ok'].mean()),
             shoulder_vis=float(d['pose'][d['pose_ok']][:, [11, 12], 3].mean()) if d['pose_ok'].any() else 0.0,
             face_inside=inside, face_width_frac=fw, second_face=second_faces(f'{cd}/real.mp4'),
             av_corr_lag0=lags[0], av_best_lag=int(best), av_best_corr=lags[best], t_cut=round(t1 - t0, 1), t_lmk=round(t2 - t1, 1))
    why = []
    if T != 240: why.append(f'frames {T}')
    if q['face_ok'] < 0.95: why.append(f"face_ok {q['face_ok']:.2f}")
    if inside < 0.95: why.append(f'face leaves crop {inside:.2f}')
    if q['second_face'] > 0.1: why.append(f"second face {q['second_face']:.2f}")
    if q['pose_ok'] < 0.9: why.append(f"pose_ok {q['pose_ok']:.2f}")
    q['drop'] = '; '.join(why)
    meta['qc'] = q
    json.dump(meta, open(f'{cd}/meta.json', 'w'), indent=1)
    json.dump(q, open(f'{cd}/qc.tmp', 'w')); os.replace(f'{cd}/qc.tmp', f'{cd}/qc.json')

for cid in sys.argv[1:]:
    try: run(cid)
    except Exception as e: print('ERR', cid, e, flush=True); open(f'{D}/work/errors.txt', 'a').write(f'{cid} {e}\n')
