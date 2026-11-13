"""Torso-up static 4:5 crop for one 240-frame window, from the Seamless COCO-WholeBody keypoints (133 x (x, y, conf), source px).
Indices: 0 nose, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 23-90 face68 (31 = chin, 40-49 brows),
91-111 left hand, 112-132 right hand.

Rule (v-final in RULE below; see data/REPORT.md):
  Hf    = median(chin_y - brow_y)                                  face height brow->chin
  top   = p2 over frames of (brow_y - TOP_K * Hf)                    a little above the hair (p5 = head's highest)
  bot   = max(median hip_y, median lowest-wrist/hand_y) + BOT_K*Hf   just below hands at rest (~hip level)
  H     = bot - top; W = 0.8 H
  W    >= hands' horizontal extent (p97 of hand/wrist x on both sides of the shoulder centre) + margin   (gesturing)
  W    >= MIN_SW * shoulder width
  if W grew, H = 1.25 W and the extra height goes below (top stays)
  centre x = median shoulder midpoint (shifted by <= 0.08 W if the gesture extent is lopsided)
  W clamped to [512/1.4, min(1080, 1920*0.8, 512/0.5)], rounded to a multiple of 4; x/y clamped inside 1080x1920.
"""
import numpy as np

TOP_K = 1.25    # hair top ~0.7 Hf above the brows + margin (1.0 cut big hair, P0845)
BOT_K = 0.35
MIN_SW = 2.0
SIDE_M = 0.06   # horizontal margin beyond the hand extent, fraction of W
SRC_W, SRC_H = 1080, 1920
OUT_W, OUT_H = 512, 640


def crop(kp, vb=None, conf=0.3):
    """kp: (240,133,3) window. Returns dict(x,y,w,h, scale, info...) or None if the body is not found."""
    c = kp[:, :, 2]
    ok = (c[:, 5] > conf) & (c[:, 6] > conf) & (c[:, 0] > conf)
    if vb is not None: ok &= vb > 0
    if ok.mean() < 0.5: return None
    k = kp[ok]
    brow = np.median(k[:, 40:50, 1], 1); chin = k[:, 31, 1]
    Hf = float(np.median(chin - brow))
    top = float(np.percentile(brow - TOP_K * Hf, 2))
    sh_mid = (k[:, 5, 0] + k[:, 6, 0]) / 2; cx = float(np.median(sh_mid))
    SW = float(np.median(np.abs(k[:, 5, 0] - k[:, 6, 0])))
    hipc = (k[:, 11, 2] > conf) & (k[:, 12, 2] > conf)
    hip_y = float(np.median((k[hipc, 11, 1] + k[hipc, 12, 1]) / 2)) if hipc.mean() > 0.3 else None
    # hands: wrist + hand keypoints with conf
    hpts = np.concatenate([k[:, [9, 10]], k[:, 91:133]], 1)          # (n, 44, 3)
    hv = hpts[:, :, 2] > conf
    low = np.where(hv, hpts[:, :, 1], -1).max(1)                        # lowest hand point per frame
    low_y = float(np.median(low[low > 0])) if (low > 0).mean() > 0.3 else None
    cands = [v for v in (hip_y, low_y) if v is not None]
    if not cands: return None
    bot = max(cands) + BOT_K * Hf
    H = bot - top; W = 0.8 * H
    # gesture extent around the shoulder centre
    xs = np.where(hv, hpts[:, :, 0], np.nan)
    lft = cx - np.nanmin(xs, 1); rgt = np.nanmax(xs, 1) - cx
    lft = float(np.nanpercentile(lft, 97)) if np.isfinite(lft).any() else 0.0
    rgt = float(np.nanpercentile(rgt, 97)) if np.isfinite(rgt).any() else 0.0
    W = max(W, MIN_SW * SW)
    for _ in range(4):   # centre may shift <= 0.08 W toward the wider gesture side; grow W until both sides fit
        shift = float(np.clip((rgt - lft) / 2, -0.08 * W, 0.08 * W))
        need = 2 * (max(lft + shift, rgt - shift) + SIDE_M * W)
        if need <= W: break
        W = need
    wmin, wmax = OUT_W / 1.4, min(SRC_W, SRC_H * 0.8, OUT_W / 0.5)
    W = float(np.clip(W, wmin, wmax))
    W = int(round(W / 4)) * 4; Hh = W * 5 // 4
    shift = float(np.clip((rgt - lft) / 2, -0.08 * W, 0.08 * W))
    x = int(round(np.clip(cx + shift - W / 2, 0, SRC_W - W)))
    y = int(round(np.clip(top, 0, SRC_H - Hh)))
    return dict(x=x, y=y, w=W, h=Hh, scale=round(OUT_W / W, 4), Hf=round(Hf, 1), SW=round(SW, 1),
                top_raw=round(top, 1), bot_raw=round(bot, 1), hand_ext=[round(lft, 1), round(rgt, 1)],
                head_cut=bool(top < -1), body_ok=float(ok.mean()))
