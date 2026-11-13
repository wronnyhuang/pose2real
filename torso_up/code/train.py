import argparse, os, time, json, random
import torch, torch.nn.functional as F, lpips, cv2, numpy as np
import data as D
from p2r import P2R, heatmap, HMC, LIPS, LM_PAST

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--size', default='512x640')         # WxH, multiples of 64; frames on disk are 512x640 and get resized
ap.add_argument('--backbone', default='sd-turbo')
ap.add_argument('--inj', default='noise')            # noise | early | controlnet | xattn
ap.add_argument('--enc', default='splat')            # splat | raster | vec
ap.add_argument('--hands', default='tokens')         # tokens | none (no hand tokens / bones / heat-map channels anywhere)
ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--audio', default='loc')           # loc (local conv feats, honest lookahead) | ctx (HuBERT L9, sees all) | win
ap.add_argument('--au_la', type=int, default=5)     # audio lookahead in video frames (30 fps); -1 = random per step
ap.add_argument('--lm_la', type=int, default=0)     # landmark lookahead in frames; -1 = random per step
ap.add_argument('--p_hide', type=float, default=0.5)  # prob. of hiding lower-face landmarks (lips must come from audio)
ap.add_argument('--no_audio', action='store_true')
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--full_ft', action='store_true')
ap.add_argument('--bs', type=int, default=4)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--minutes', type=float, default=35)
ap.add_argument('--max_steps', type=int, default=10 ** 6)
ap.add_argument('--gan_start', type=int, default=600)
ap.add_argument('--w_gan', type=float, default=0.5)
ap.add_argument('--w_mouth', type=float, default=3.0)
ap.add_argument('--w_hand', type=float, default=2.0)   # hand-crop LPIPS + L1 (0 disables)
ap.add_argument('--hand_gan', action='store_true')     # second patch discriminator on the 128 px hand crops
ap.add_argument('--w_hand_gan', type=float, default=0.25)
ap.add_argument('--p_hand_frame', type=float, default=0.5)  # prob. a sample comes from a frame with >= 1 visible hand
ap.add_argument('--w_xy', type=float, default=20.0)
ap.add_argument('--splits', default='train')
ap.add_argument('--sets', default='')                # comma list of manifest `set`s (prior,gesture,...); '' = all
ap.add_argument('--max_clips', type=int, default=0)
ap.add_argument('--resume', default='')
ap.add_argument('--no_ckpt', action='store_true')    # q.sh passes it (no UNet gradient checkpointing) ...
ap.add_argument('--ckpt', action='store_true')       # ... and a queue line can turn checkpointing back on (24 GB cards)
ap.add_argument('--refbase', action='store_true')
ap.add_argument('--w_lips', type=float, default=1.0)
ap.add_argument('--enc_init', default='')   # conditioning encoder pre-trained by train_xy.py
ap.add_argument('--boost', type=int, default=0)   # extra copies of heldout_ref clips (seen-identity setting)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--dev', default='cuda')
ap.add_argument('--workers', type=int, default=12)
ap.add_argument('--amp', default='bf16')        # bf16 | none (fp32; CPU smoke tests on CPUs without bf16 hardware)
ap.add_argument('--save_every', type=int, default=1000)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = a.dev; dt = dev.split(':')[0]
W, H = map(int, a.size.lower().split('x')); assert W % 64 == 0 and H % 64 == 0, a.size
if dt == 'cuda': torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

clips = D.ready_ids(a.splits.split(','), a.audio, a.sets.split(',') if a.sets else None)
if a.boost: clips = clips + [c for c in clips if c['split'] == 'heldout_ref'] * a.boost
if a.max_clips: clips = clips[:a.max_clips]
assert clips, 'no ready clips (run prep.py)'
print('train clips', len(clips), 'identities', len({c['identity'] for c in clips}), 'size', W, H, flush=True)
low = D.lower_face_idx(clips)
print('lower-face points hidden in audio-lips mode:', int(low.sum()), flush=True)
adim = 512 if a.audio == 'loc' else 768
LAI = {0: 0, 2: 1, 5: 2, 10: 3}
HANDS = a.hands == 'tokens'
model = P2R(a.backbone, a.inj, a.enc, a.depth, adim, a.rank, low, full_ft=a.full_ft, refbase=a.refbase, hands=a.hands, size=(W, H)).to(dev)
model.set_ckpt(a.ckpt or not a.no_ckpt); model.train(); model.vae.eval()
if a.resume: model.load_trainable(torch.load(a.resume, map_location='cpu'))
if a.enc_init:
    sd = torch.load(a.enc_init, map_location='cpu'); cur = model.cond.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k].shape and not k.startswith('head')}
    print('enc_init: loaded', len(sd), 'of', len(cur), 'tensors', flush=True)
    model.cond.load_state_dict(sd, strict=False)
from common_gan import MultiD, SmallD, d_hinge, g_hinge
Dn = MultiD(3 + HMC).to(dev)
Dh = SmallD(3).to(dev) if a.hand_gan else None
lp = lpips.LPIPS(net='vgg', verbose=False).to(dev).eval().requires_grad_(False)
enc_p = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith('cond.') or n.startswith('ref_'))]
unet_p = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith('cond.') or n.startswith('ref_'))]
optG = torch.optim.AdamW([{'params': enc_p, 'lr': a.lr * 2}, {'params': unet_p, 'lr': a.lr if not a.full_ft else a.lr * 0.2}], weight_decay=1e-2)
optD = torch.optim.AdamW(list(Dn.parameters()) + (list(Dh.parameters()) if Dh else []), lr=a.lr, betas=(0.5, 0.999))
print('trainable M: encoder %.1f unet %.1f' % (sum(p.numel() for p in enc_p) / 1e6, sum(p.numel() for p in unet_p) / 1e6), flush=True)
cfg = vars(a).copy(); cfg['enc_params_M'] = sum(p.numel() for p in enc_p) / 1e6; cfg['clips'] = len(clips)
json.dump(cfg, open(a.out + '/cfg.json', 'w'))

ds = D.DS(clips, a.audio, size=(W, H), p_hand=a.p_hand_frame)   # same sampling for --hands none (clean ablation)
it = iter(torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, drop_last=True,
                                      prefetch_factor=4 if a.workers else None, persistent_workers=a.workers > 0))
PX = torch.tensor([W, H], dtype=torch.float32, device=dev)


def crops(imgs, boxes, out=128):
    """boxes: list of (i, x, y, side) in pixels -> per image tensor a (len(boxes),C,out,out) stack of resized crops."""
    return [torch.cat([F.interpolate(im[i:i + 1, :, y:y + s, x:x + s], size=(out, out), mode='bilinear', align_corners=False)
                       for i, x, y, s in boxes]) for im in imgs]


def box(cx, cy, side):
    s = int(round(min(side, W, H)))
    return int(round(min(max(cx - s / 2, 0), W - s))), int(round(min(max(cy - s / 2, 0), H - s))), s


def mouth_boxes(face):
    """Square box around the lips (TRUE landmarks), side 1.6 x inter-ocular (outer eye corners) in px, min 32."""
    f = face[..., :2].float() * PX
    c = f[:, LIPS].mean(1); io = (f[:, 263] - f[:, 33]).norm(dim=-1)
    return [(i, *box(c[i, 0].item(), c[i, 1].item(), max(32.0, 1.6 * io[i].item()))) for i in range(f.shape[0])]


def hand_boxes(hands):
    """One square box per visible hand (TRUE landmarks): side 1.6 x max(bbox w, h) in px, min 40, clamped to the frame."""
    bx = []
    for i in range(hands.shape[0]):
        for h in range(2):
            if hands[i, h, 0, 3] < 0.5: continue
            p = hands[i, h, :, :2].float() * PX
            lo, hi = p.min(0).values, p.max(0).values; c = (lo + hi) / 2
            if not (0 <= c[0] < W and 0 <= c[1] < H): continue       # pushed out of frame by the augmentation
            bx.append((i, *box(c[0].item(), c[1].item(), max(40.0, 1.6 * (hi - lo).max().item()))))
    return bx


def t2im(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


def zero(): return torch.zeros((), device=dev)


t0 = time.time(); step = 0; log = open(a.out + '/log.txt', 'a'); acc = {}; nh_acc = 0
while step < a.max_steps and time.time() - t0 < a.minutes * 60:
    b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
    if a.no_audio: b['audio'] = torch.zeros_like(b['audio'])
    y = b['y']; B = y.shape[0]
    hide = torch.rand(B, device=dev) < a.p_hide
    with torch.autocast(dt, dtype=torch.bfloat16, enabled=a.amp == 'bf16'):
        ref_lat = model.encode(b['ref'])
        noise = torch.randn(B, 4, H // 8, W // 8, device=dev)
        au_la = a.au_la if a.au_la >= 0 else random.choice([0, 2, 5, 10])
        lm_la = a.lm_la if a.lm_la >= 0 else random.choice([0, 0, 2, 5])
        if a.audio == 'win': b['audio'] = b['audio'][:, LAI[au_la]]
        o = model(b, noise, ref_lat, lm_la=lm_la, au_la=au_la, mask_lower=hide)
        out = o['img'].float()
        l2 = F.mse_loss(out, y)
        llp = lp(out, y).mean()
        face_t = b['face_w'][:, LM_PAST]; hands_t = b['hands_w'][:, LM_PAST]
        mo, my = crops([out, y], mouth_boxes(face_t))
        lm = lp(mo, my).mean() + F.l1_loss(mo, my)
        hb = hand_boxes(hands_t) if (a.w_hand > 0 or Dh is not None) else []
        nh_acc += len(hb)
        lh = zero(); ho = hy = None
        if hb:
            ho, hy = crops([out, y], hb)
            lh = lp(ho, hy).mean() + F.l1_loss(ho, hy)
        hm = heatmap(face_t, b['pose_w'][:, LM_PAST], hands_t if HANDS else None, H // 4, W // 4)
        hm_up = F.interpolate(hm, size=(H, W), mode='bilinear', align_corners=False)
        lxy = zero()
        if hide.any() and a.enc != 'raster':
            # error in units of the inter-ocular distance (px) x 0.2: scale-free, ~ PRIOR's magnitude for its 384 bust frames
            h = o['hide']; NFP = low.shape[0]; h = h[:, :NFP]
            wl = torch.ones(NFP, device=dev); wl[LIPS] = a.w_lips
            xt = o['xy_true'][:, :NFP].float() * PX; xp = o['xy_pred'][:, :NFP].float() * PX
            io = (xt[:, 263] - xt[:, 33]).norm(dim=-1).clamp(min=4).view(B, 1)
            lxy = ((xp - xt).abs().sum(-1) / io * 0.2 * wl)[h].sum() / (wl.expand_as(h)[h].sum())
        loss = l2 + 5 * llp + a.w_mouth * lm + a.w_hand * lh + a.w_xy * lxy
        lg = zero(); lgh = zero()
        if step >= a.gan_start:
            lg = g_hinge(Dn, torch.cat([hm_up, out], 1))
            loss = loss + a.w_gan * lg
            if Dh is not None and hb:
                lgh = g_hinge(Dh, ho)
                loss = loss + a.w_hand_gan * lgh
    optG.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
    optG.step()
    ld = zero(); ldh = zero()
    if step >= a.gan_start - 200:
        with torch.autocast(dt, dtype=torch.bfloat16, enabled=a.amp == 'bf16'):
            ld = d_hinge(Dn, torch.cat([hm_up, y], 1), torch.cat([hm_up, out.detach()], 1))
            if Dh is not None and hb: ldh = d_hinge(Dh, hy, ho.detach())
        optD.zero_grad(set_to_none=True); (ld + ldh).backward(); optD.step()
    for k, v in (('l2', l2), ('lpips', llp), ('mouth', lm), ('hand', lh), ('xy', lxy), ('g', lg), ('d', ld), ('gh', lgh), ('dh', ldh)):
        acc[k] = acc.get(k, 0) + float(v)
    step += 1
    if step % 50 == 0 or step <= 2:
        n = 50 if step % 50 == 0 else 1
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dt == 'cuda' else 0
        msg = f'step {step} t {time.time() - t0:.0f}s mem {mem:.1f}G hands/batch {nh_acc / n:.2f} ' + ' '.join(f'{k} {v / n:.4f}' for k, v in acc.items())
        print(msg, flush=True); log.write(msg + '\n'); log.flush(); acc = {}; nh_acc = 0
    if step % 500 == 0 or step == 20:
        hv = (hm_up[:, [0, 6, 9]].clamp(0, 1) * 2 - 1)     # lips, pose bones, hand bones
        grid = np.concatenate([np.concatenate(list(t2im(z)), 0) for z in (hv, b['ref'], out, y)], 1)
        cv2.imwrite(f'{a.out}/sample_{step:06d}.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if step % a.save_every == 0:
        torch.save(model.trainable_state(), a.out + '/last.pt')
torch.save(model.trainable_state(), a.out + '/last.pt')
cfg.update(steps=step, train_minutes=(time.time() - t0) / 60)
json.dump(cfg, open(a.out + '/cfg.json', 'w'))
print('DONE', step, flush=True)
