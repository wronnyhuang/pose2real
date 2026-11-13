"""Shared landmark extractor for pose2real (contract: ROOT/CONTRACT.md).

extract(video_path) -> dict of numpy arrays, one row per frame:
  face    (T,478,3) float32  FaceLandmarker x,y normalised [0,1], z as MediaPipe gives (NaN-free: zeros when missing)
  face_ok (T,) bool
  pose    (T,33,4) float32   PoseLandmarker x,y (normalised), z, visibility (zeros when missing)
  pose_ok (T,) bool
  bs      (T,52) float32     face blendshapes (zeros when face missing)

CLI: python lmk.py in.mp4 out.npz

Deterministic: VIDEO running mode, fixed model files in ROOT/common/models/, fresh landmarker
instances per call, timestamps = round(frame_index * 1000 / fps), CPU delegate.
Missing detections are stored as zeros with *_ok False (never NaN).
"""
import os, sys
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models')
FACE_MODEL = os.path.join(MODELS, 'face_landmarker.task')
POSE_VARIANT = 'heavy'   # chosen 11-12 on 4 test clips: lowest shoulder jitter; lite/full kept in models/ for reference
POSE_MODEL = os.path.join(MODELS, f'pose_landmarker_{POSE_VARIANT}.task')
FACE_CONF = 0.4
POSE_CONF = 0.4


def _make():
    base = lambda p: mpt.BaseOptions(model_asset_path=p, delegate=mpt.BaseOptions.Delegate.CPU)
    face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(
        base_options=base(FACE_MODEL), running_mode=V.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True, output_facial_transformation_matrixes=False,
        min_face_detection_confidence=FACE_CONF, min_face_presence_confidence=FACE_CONF,
        min_tracking_confidence=FACE_CONF))
    pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(
        base_options=base(POSE_MODEL), running_mode=V.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=POSE_CONF, min_pose_presence_confidence=POSE_CONF,
        min_tracking_confidence=POSE_CONF, output_segmentation_masks=False))
    return face, pose


def extract(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'cannot open {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    T = len(frames)
    face = np.zeros((T, 478, 3), np.float32); face_ok = np.zeros(T, bool)
    pose = np.zeros((T, 33, 4), np.float32); pose_ok = np.zeros(T, bool)
    bs = np.zeros((T, 52), np.float32)
    fl, pl = _make()
    try:
        for t, f in enumerate(frames):
            ts = int(round(t * 1000.0 / fps))
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            r = fl.detect_for_video(img, ts)
            if r.face_landmarks:
                lm = r.face_landmarks[0]
                face[t] = np.array([[p.x, p.y, p.z] for p in lm[:478]], np.float32)
                face_ok[t] = True
                if r.face_blendshapes:
                    bs[t] = np.array([b.score for b in r.face_blendshapes[0]][:52], np.float32)
            r = pl.detect_for_video(img, ts)
            if r.pose_landmarks:
                pose[t] = np.array([[p.x, p.y, p.z, p.visibility] for p in r.pose_landmarks[0]], np.float32)
                pose_ok[t] = True
    finally:
        fl.close(); pl.close()
    return dict(face=face, face_ok=face_ok, pose=pose, pose_ok=pose_ok, bs=bs)


def save(d, out_path):
    np.savez_compressed(out_path, **d)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python lmk.py in.mp4 out.npz'); sys.exit(2)
    d = extract(sys.argv[1])
    save(d, sys.argv[2])
    T = len(d['face_ok'])
    print(f'{sys.argv[1]}: T={T} face_ok={d["face_ok"].mean():.3f} pose_ok={d["pose_ok"].mean():.3f}')
