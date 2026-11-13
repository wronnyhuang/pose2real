"""Shared landmark extractor for torso_up (contract: ROOT/CONTRACT.md). Face + pose as pose2real, plus HANDS.

extract(video_path) -> dict of numpy arrays, one row per frame:
  face        (T,478,3) float32  FaceLandmarker x,y normalised [0,1], z as MediaPipe gives (zeros when missing)
  face_ok     (T,) bool
  pose        (T,33,4) float32   PoseLandmarker (heavy) x,y (normalised, may leave [0,1]), z, visibility
  pose_ok     (T,) bool
  bs          (T,52) float32     face blendshapes (zeros when face missing)
  hands       (T,2,21,3) float32 HandLandmarker 21 points, x,y normalised to the FULL frame, z as MediaPipe gives.
                                 index 0 = the person's anatomical LEFT hand, 1 = RIGHT (zeros when missing)
  hands_ok    (T,2) bool
  hands_score (T,2) float32      handedness-classifier score of the detection (0 when missing; HandLandmarker exposes no
                                 separate presence score; every kept hand passed detection/presence >= HAND_CONF)
CLI: python lmk.py in.mp4 out.npz

HAND METHOD (chosen 11-13 on 8 torso-up 512x640 crops of 8 identities, data/work/handtest/; 1,920 frames, 3,840 hand slots):
  HandLandmarker float16 (models/hand_landmarker.task, sha256 fbc2a300...), IMAGE mode (palm detection on EVERY frame, no
  tracking state), num_hands=2, on the full 512x640 frame, detection/presence conf 0.3.
  Left/right: each detected hand goes to the nearer pose wrist (pose 15 = left, 16 = right; greedy on wrist-to-wrist pixel
  distance); a hand whose wrist is > HAND_GATE x shoulder width from that pose wrist is dropped. Frames without a pose fall
  back to MediaPipe's handedness label; measured with this mediapipe build on these (non-mirrored) crops, label "Left" = the
  person's anatomical LEFT hand in 119/120 pose-assigned detections.
  Measured (mean over the 8 clips; det = share of all hand slots detected, incl. one hand in a pocket for a whole clip, so the
  ceiling is ~0.95; jitter = median / p90 per-frame 2nd difference of the 21 points / hand size (wrist->middle MCP);
  wd = hand wrist to pose wrist / shoulder width, median / p99):
    full frame, VIDEO mode (tracking)       det 0.861  jitter 0.020/0.095  wd 0.04/0.11   (lost a clasped 2nd hand for a whole clip)
    full frame x2 upscale, VIDEO mode       det 0.858  jitter 0.021/0.106  wd 0.04/0.11
    full frame, IMAGE mode       <- CHOSEN  det 0.934  jitter 0.020/0.083  wd 0.04/0.13
    wrist ROI 1.3*forearm|0.55*SW, 256 px   det 0.918  jitter 0.028/0.174  wd 0.09/0.23   (clasped hands: ROI grabs the wrong hand)
    wrist ROI 1.0*forearm|0.45*SW, 256 px   det 0.836  jitter 0.049/0.281  wd 0.08/0.23
  IMAGE mode on the full frame is per-frame independent (no tracker that can latch onto or lose a hand), so it behaves the same on
  generated video and is deterministic. Typical hand size at 512x640: wrist->middle-MCP ~43 px (whole hand ~80-90 px).
  Speed: ~23 s / 240 frames for hands, ~31 s pose heavy, ~10 s face (1 thread each).

Deterministic: fixed model files in ROOT/common/models/, fresh landmarker instances per call, face/pose in VIDEO mode with
timestamps = round(frame_index * 1000 / fps), hands in IMAGE mode, CPU delegate. Missing detections are zeros (never NaN).
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
POSE_VARIANT = 'heavy'
POSE_MODEL = os.path.join(MODELS, f'pose_landmarker_{POSE_VARIANT}.task')
HAND_MODEL = os.path.join(MODELS, 'hand_landmarker.task')
FACE_CONF = 0.4
POSE_CONF = 0.4
HAND_CONF = 0.3
HAND_GATE = 0.6   # max distance hand wrist <-> assigned pose wrist, in shoulder widths


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
    hand = V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(
        base_options=base(HAND_MODEL), running_mode=V.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=HAND_CONF, min_hand_presence_confidence=HAND_CONF,
        min_tracking_confidence=HAND_CONF))
    return face, pose, hand


def assign_hands(res, pose_row, pose_ok, W, H):
    """-> hands (2,21,3), ok (2,), score (2,) from one HandLandmarker result, sides by nearest pose wrist."""
    hands = np.zeros((2, 21, 3), np.float32); ok = np.zeros(2, bool); sc = np.zeros(2, np.float32)
    if not res.hand_landmarks:
        return hands, ok, sc
    cands = [(np.array([[p.x, p.y, p.z] for p in lm[:21]], np.float32), float(res.handedness[i][0].score),
              res.handedness[i][0].category_name) for i, lm in enumerate(res.hand_landmarks)]
    if pose_ok:
        px = np.array([W, H], np.float32)
        wr = pose_row[[15, 16], :2] * px
        sw = max(float(np.linalg.norm((pose_row[11, :2] - pose_row[12, :2]) * px)), 1.0)
        pairs = sorted((float(np.linalg.norm(c[0][0, :2] * px - wr[s])), s, i) for i, c in enumerate(cands) for s in (0, 1))
        used = set()
        for d, s, i in pairs:
            if i in used or ok[s] or d > HAND_GATE * sw:
                continue
            hands[s], sc[s], ok[s] = cands[i][0], cands[i][1], True
            used.add(i)
    else:
        for a, s_, lab in sorted(cands, key=lambda c: -c[1]):
            s = 0 if lab == 'Left' else 1   # measured: label 'Left' == anatomical left on non-mirrored video (119/120)
            if not ok[s]:
                hands[s], sc[s], ok[s] = a, s_, True
    return hands, ok, sc


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
    H, W = frames[0].shape[:2] if T else (1, 1)
    face = np.zeros((T, 478, 3), np.float32); face_ok = np.zeros(T, bool)
    pose = np.zeros((T, 33, 4), np.float32); pose_ok = np.zeros(T, bool)
    bs = np.zeros((T, 52), np.float32)
    hands = np.zeros((T, 2, 21, 3), np.float32); hands_ok = np.zeros((T, 2), bool); hands_score = np.zeros((T, 2), np.float32)
    fl, pl, hl = _make()
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
            hands[t], hands_ok[t], hands_score[t] = assign_hands(hl.detect(img), pose[t], pose_ok[t], W, H)
    finally:
        fl.close(); pl.close(); hl.close()
    return dict(face=face, face_ok=face_ok, pose=pose, pose_ok=pose_ok, bs=bs,
                hands=hands, hands_ok=hands_ok, hands_score=hands_score)


def save(d, out_path):
    np.savez_compressed(out_path, **d)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('usage: python lmk.py in.mp4 out.npz'); sys.exit(2)
    d = extract(sys.argv[1])
    save(d, sys.argv[2])
    T = len(d['face_ok'])
    print(f'{sys.argv[1]}: T={T} face_ok={d["face_ok"].mean():.3f} pose_ok={d["pose_ok"].mean():.3f} '
          f'hands_ok L={d["hands_ok"][:, 0].mean():.3f} R={d["hands_ok"][:, 1].mean():.3f}')
