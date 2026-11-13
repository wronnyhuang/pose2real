import sys, os, time, numpy as np
sys.path.insert(0, __import__('os').path.expandvars('$WORK/eval'))
import run_eval, lm_metrics, judge, make_refs
v = sys.argv[1]; cid = os.path.basename(v)[:-4]
m, _ = run_eval.lmk_mod()
t = time.time(); a = m.extract(v); t1 = time.time() - t; b = m.extract(v)
print('lmk secs', round(t1, 1), 'identical:', all(np.array_equal(a[k], b[k]) for k in a))
print('hands_ok', a['hands_ok'].mean(0))
np.savez(f'dev/{cid}_lm.npz', **a)
r = lm_metrics.compare(a, b); print({k: r[k] for k in ('hand_err', 'hand_det_rate', 'arm_err', 'pose_err', 'n_hand_frames')})
plan = judge.hand_plan(f'dev/{cid}_lm.npz', 512, 640); print(plan)
judge.hand_strip(v, plan, f'dev/{cid}_hands.png'); judge.grid(v, f'dev/{cid}_grid.png')
n = make_refs.blurhands(cid, v, f'dev/{cid}_blur.mp4', f'dev/{cid}_lm.npz'); print('blurred', n)
c = m.extract(f'dev/{cid}_blur.mp4'); r = lm_metrics.compare(a, c)
print({k: r[k] for k in ('hand_err', 'hand_err_hs', 'hand_det_rate', 'hand_extra_rate', 'finger_pose_err', 'finger_angle_err_deg', 'face_nme', 'pose_err', 'arm_err')})
judge.hand_strip(f'dev/{cid}_blur.mp4', plan, f'dev/{cid}_hands_blur.png')
