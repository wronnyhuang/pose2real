#!/usr/bin/env python3
"""Reference "runs" built from the real heldout clips -> eval/ref/<name>/<clip_id>.mp4
  real           symlinks to ROOT/data/clips/<id>/real.mp4                       (ceiling: landmark errors must be exactly 0)
  real_crf28     H.264 crf 28 re-encode (libx264, preset medium, yuv420p)        (compression noise floor)
  real_blurhands gaussian blur sigma 3 px over the hand regions only (conditioning hands of that frame: convex hull of the 21
                 points dilated by 0.35 x hand size, 5 px feathered edge), everything else bit-exact source pixels,
                 encoded lossless (libx264 -qp 0, yuv420p)                       (how the hand metrics react to mushy hands)
    python eval/make_refs.py [real real_crf28 real_blurhands]
"""
import os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def crf28(cid, src, dst):
    subprocess.run(['nice', '-n', '10', 'ffmpeg', '-v', 'error', '-y', '-i', src, '-an', '-c:v', 'libx264', '-crf', '28',
                    '-preset', 'medium', '-pix_fmt', 'yuv420p', '-threads', '2', dst], check=True)


def blurhands(cid, src, dst, lm):
    import cv2; cv2.setNumThreads(1)
    z = np.load(lm); hands = z['hands'][..., :2]; ok = z['hands_ok'].astype(bool)
    cap = cv2.VideoCapture(src); fr = []
    while True:
        k, f = cap.read()
        if not k: break
        fr.append(f)
    h, w = fr[0].shape[:2]
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '30',
                          '-i', '-', '-c:v', 'libx264', '-qp', '0', '-preset', 'medium', '-pix_fmt', 'yuv420p', '-threads', '2', dst],
                         stdin=subprocess.PIPE)
    n_blur = 0
    for t, f in enumerate(fr):
        m = np.zeros((h, w), np.uint8)
        for s in (0, 1):
            if t < len(ok) and ok[t, s]:
                pts = hands[t, s] * [w, h]
                hs = np.linalg.norm(pts[0] - pts[9])
                hull = cv2.convexHull(pts.astype(np.float32)).astype(np.int32)
                cv2.fillConvexPoly(m, hull, 255)
                r = max(3, int(round(0.35 * hs)))
                m = np.maximum(m, cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))))
        if m.any():
            n_blur += 1
            a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2.0)[..., None]
            b = cv2.GaussianBlur(f, (0, 0), 3.0)
            f = np.clip(f * (1 - a) + b * a, 0, 255).astype(np.uint8)
        p.stdin.write(f.tobytes())
    p.stdin.close(); p.wait()
    if p.returncode: raise RuntimeError(f'ffmpeg failed {cid}')
    return n_blur


def one(args):
    name, cid = args
    import run_eval
    inp = run_eval.inputs(cid)
    d = os.path.join(HERE, 'ref', name); os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, cid + '.mp4')
    if name == 'real':
        if os.path.lexists(dst): os.remove(dst)
        os.symlink(inp['real'], dst); return cid, 'link'
    tmp = dst + '.tmp.mp4'
    if name == 'real_crf28': crf28(cid, inp['real'], tmp); info = 'crf28'
    else: info = f'blurred frames {blurhands(cid, inp["real"], tmp, inp["lm"])}'
    os.replace(tmp, dst); return cid, info


def main():
    names = sys.argv[1:] or ['real', 'real_crf28', 'real_blurhands']
    os.nice(10)
    import run_eval
    ids = [c for c, _ in run_eval.heldout()]
    with ProcessPoolExecutor(10) as ex:
        for cid, info in ex.map(one, [(n, c) for n in names for c in ids]): print(cid, info, flush=True)


if __name__ == '__main__':
    main()
