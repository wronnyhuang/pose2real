#!/usr/bin/env python3
"""Unit tests of lm_metrics on synthetic arrays:  python eval/test_metrics.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm_metrics as L

rng = np.random.default_rng(0)
T, W, H = 60, 512, 640


def synth():
    d = dict(face=np.zeros((T, 478, 3), np.float32), face_ok=np.ones(T, bool), pose=np.zeros((T, 33, 4), np.float32),
             pose_ok=np.ones(T, bool), bs=np.zeros((T, 52), np.float32), hands=np.zeros((T, 2, 21, 3), np.float32),
             hands_ok=np.zeros((T, 2), bool), hands_score=np.zeros((T, 2), np.float32))
    d['face'][:, :, :2] = 0.4 + 0.2 * rng.random((T, 478, 2))
    d['face'][:, 33, :2] = [0.45, 0.3]; d['face'][:, 263, :2] = [0.55, 0.3]
    d['pose'][:, :, :2] = 0.2 + 0.6 * rng.random((T, 33, 2)); d['pose'][:, :, 3] = 0.9
    d['pose'][:, 11, :2] = [0.3, 0.5]; d['pose'][:, 12, :2] = [0.7, 0.5]       # shoulder width = 0.4*512 = 204.8 px
    base = rng.random((21, 2)) * [0.08, 0.064]                                  # ~40 px hand
    base[0] = [0, 0]; base[9] = [0.0, -0.05]                                    # hand size = 0.05*640 = 32 px
    d['hands'][:, 0, :, :2] = base + [0.3, 0.7]; d['hands'][:, 1, :, :2] = base + [0.6, 0.7]
    d['hands_ok'][:40, 0] = True; d['hands_ok'][10:, 1] = True
    return d


def copy(d): return {k: v.copy() for k, v in d.items()}


c = synth()
r = L.compare(c, copy(c), W, H)
for k in ('face_nme', 'mouth_nme', 'pose_err', 'shoulder_err', 'elbow_err', 'wrist_err', 'arm_err', 'hand_err', 'hand_err_hs',
          'hand_err_pen', 'finger_pose_err', 'finger_angle_err_deg'):
    assert r[k] == 0, (k, r[k])
assert r['hand_det_rate'] == 1 and r['hand_extra_rate'] == 0 and r['det_rate'] == 1
assert r['n_hand_frames'] == 40 + 50 and r['n_nohand_frames'] == 120 - 90
assert abs(r['shoulder_width_px'] - 204.8) < 0.01

# similarity transform of every generated hand (in PIXELS) -> finger_pose_err 0, angle 0, hand_err > 0
g = copy(c)
th = np.radians(25); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
px = g['hands'][..., :2] * [W, H]
ctr = px.mean(2, keepdims=True)
px = (px - ctr) @ R.T * 1.3 + ctr + [10, -5]
g['hands'][..., :2] = px / [W, H]
r = L.compare(c, g, W, H)
assert r['finger_pose_err'] < 1e-3 and r['finger_angle_err_deg'] < 1e-3 and r['hand_err'] > 1, r

# pure translation of 12.8 px in x for wrist 15 -> wrist_err = 12.8/204.8*100 = 6.25 (only one of 2 wrists) -> mean 3.125
g = copy(c); g['pose'][:, 15, 0] += 12.8 / W
r = L.compare(c, g, W, H)
assert abs(r['wrist_err'] - 3.125) < 1e-3 and r['elbow_err'] == 0 and abs(r['arm_err'] - 12.8 / 4 / 204.8 * 100) < 1e-3, r
# non-square check: 12.8 px in y
g = copy(c); g['pose'][:, 15, 1] += 12.8 / H
assert abs(L.compare(c, g, W, H)['wrist_err'] - 3.125) < 1e-3

# translate a hand by 16 px in y: hand_err = 16/204.8*100 = 7.8125, hand_err_hs = 16/32*100 = 50, shape 0
g = copy(c); g['hands'][:, :, :, 1] += 16 / H
r = L.compare(c, g, W, H)
assert abs(r['hand_err'] - 7.8125) < 1e-3 and abs(r['hand_err_hs'] - 50) < 1e-3 and r['finger_pose_err'] < 1e-3, r

# misses / extras
g = copy(c); g['hands_ok'][:20, 0] = False; g['hands_ok'][:5, 1] = True; g['hands'][:20, 0] = 0
r = L.compare(c, g, W, H)
assert abs(r['hand_det_rate'] - 70 / 90) < 1e-3 and abs(r['hand_extra_rate'] - 5 / 30) < 1e-3, r
assert abs(r['hand_err_pen'] - 20 * 100 / 90) < 1e-3 and r['hand_err'] == 0, r

# bent finger: rotate the index tip (8) about DIP (7) by 30 deg -> angle error at DIP = 30 deg on 1 of 15 joints
g = copy(c)
P = g['hands'][..., :2] * [W, H]
v = P[:, :, 8] - P[:, :, 7]; th = np.radians(30); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
P[:, :, 8] = P[:, :, 7] + v @ R.T
g['hands'][..., :2] = P / [W, H]
r = L.compare(c, g, W, H)
assert abs(r['finger_angle_err_deg'] - 2.0) < 1e-2 and r['finger_pose_err'] > 0, r

# run-mean weighting
m = L.hand_run_mean([dict(hands_available=True, n_hand_frames=10, n_nohand_frames=0, n_hand_both=10, hand_det_rate=1.0,
                          hand_err=2.0, hand_err_pen=2.0, hand_err_hs=1, hand_err_hs_pen=1, finger_pose_err=1, finger_angle_err_deg=1),
                     dict(hands_available=True, n_hand_frames=30, n_nohand_frames=10, n_hand_both=15, hand_det_rate=0.5,
                          hand_extra_rate=0.1, hand_err=4.0, hand_err_pen=52.0, hand_err_hs=1, hand_err_hs_pen=1,
                          finger_pose_err=1, finger_angle_err_deg=1),
                     dict(hands_available=True, n_hand_frames=0, n_nohand_frames=480, n_hand_both=0, hand_extra_rate=0.0)])
assert abs(m['hand_det_rate'] - 0.625) < 1e-6 and abs(m['hand_err'] - 3.2) < 1e-6 and abs(m['hand_extra_rate'] - 0.1 / 49) < 1e-4, m
print('all lm_metrics tests passed')
