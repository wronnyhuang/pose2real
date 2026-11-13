import sys, os, time, numpy as np
v=sys.argv[1]; os.environ['LMK_POSE_VARIANT']=v
sys.path.insert(0,__import__('os').path.expandvars('$WORK/common')); import lmk
for cid in sys.argv[2:]:
    t=time.time(); d=lmk.extract(__import__('os').path.expandvars(f'$WORK/data/pairs/{cid}/real.mp4')); dt=time.time()-t
    p=d['pose']; ok=d['pose_ok']
    sh=p[ok][:, [11,12], :]
    vis=sh[...,3].mean() if ok.any() else 0
    jit=np.median(np.abs(np.diff(sh[...,:2],axis=0))) if ok.sum()>2 else -1
    nose=np.median(np.abs(p[ok][:,0,:2]-d['face'][ok][:,1,:2])) if ok.any() else -1
    print(v,cid,f'{dt:.0f}s face_ok={d["face_ok"].mean():.3f} pose_ok={ok.mean():.3f} shoulder_vis={vis:.2f} sh_jitter={jit*384:.2f}px nose-facenose={nose*384:.1f}px sh_y={sh[...,1].mean():.2f}',flush=True)
