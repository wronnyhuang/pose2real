import argparse, os, time, json, random
import torch, torch.nn as nn, torch.nn.functional as F, lpips, cv2, numpy as np
import data as D
from p2r import P2R, heatmap, LIPS, LM_PAST, NF

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--backbone', default='sd-turbo')
ap.add_argument('--inj', default='noise')
ap.add_argument('--enc', default='splat')
ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--audio', default='loc')           # loc (local conv feats, honest lookahead) | ctx (HuBERT L9, sees all)
ap.add_argument('--au_la', type=int, default=5)     # audio lookahead in video frames (30 fps)
ap.add_argument('--lm_la', type=int, default=0)     # landmark lookahead in frames
ap.add_argument('--p_hide', type=float, default=0.5)  # prob. of hiding lower-face landmarks (lips must come from audio)
ap.add_argument('--no_audio', action='store_true')
ap.add_argument('--rank', type=int, default=32)
ap.add_argument('--full_ft', action='store_true')
ap.add_argument('--bs', type=int, default=6)
ap.add_argument('--lr', type=float, default=1e-4)
ap.add_argument('--minutes', type=float, default=35)
ap.add_argument('--max_steps', type=int, default=10 ** 6)
ap.add_argument('--gan_start', type=int, default=600)
ap.add_argument('--w_gan', type=float, default=0.5)
ap.add_argument('--w_mouth', type=float, default=3.0)
ap.add_argument('--w_xy', type=float, default=20.0)
ap.add_argument('--splits', default='train')
ap.add_argument('--max_clips', type=int, default=0)
ap.add_argument('--resume', default='')
ap.add_argument('--no_ckpt', action='store_true')
ap.add_argument('--origin', default='render2real')   # render2real = the 126 seed train clips only (all ablations); all = + expansion
ap.add_argument('--refbase', action='store_true')
ap.add_argument('--w_lips', type=float, default=1.0)
ap.add_argument('--enc_init', default='')   # conditioning encoder pre-trained by train_xy.py
ap.add_argument('--boost', type=int, default=0)   # extra copies of heldout_ref clips (seen-identity setting)
ap.add_argument('--w_sync', type=float, default=0.0)   # SyncNet expert loss; batches become 5-frame windows (bs must be a multiple of 5)
ap.add_argument('--sync_mouth', action='store_true')
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

clips = D.ready_ids(a.splits.split(','), a.audio)
if a.origin != 'all': clips = [c for c in clips if c.get('origin', 'render2real') == a.origin]
if a.boost: clips = clips + [c for c in clips if c['split'] == 'heldout_ref'] * a.boost
if a.max_clips: clips = clips[:a.max_clips]
print('train clips', len(clips), 'identities', len({c['identity'] for c in clips}), flush=True)
low = D.lower_face_idx([c for c in clips if True])
print('lower-face points hidden in audio-lips mode:', int(low.sum()), flush=True)
adim = 512 if a.audio == 'loc' else 768
LAI = {0: 0, 2: 1, 5: 2, 10: 3}
model = P2R(a.backbone, a.inj, a.enc, a.depth, adim, a.rank, low, full_ft=a.full_ft, refbase=a.refbase).to(dev)
model.set_ckpt(not a.no_ckpt); model.train(); model.vae.eval()
if a.resume: model.load_trainable(torch.load(a.resume))
if a.enc_init:
    sd = torch.load(a.enc_init); cur = model.cond.state_dict()
    sd = {k: v for k, v in sd.items() if k in cur and v.shape == cur[k].shape and not k.startswith('head')}
    print('enc_init: loaded', len(sd), 'of', len(cur), 'tensors', flush=True)
    model.cond.load_state_dict(sd, strict=False)
from common_gan import MultiD, d_hinge, g_hinge
Dn = MultiD(3 + 8).to(dev)
lp = lpips.LPIPS(net='vgg').to(dev).eval().requires_grad_(False)
enc_p = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith('cond.') or n.startswith('ref_'))]
unet_p = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith('cond.') or n.startswith('ref_'))]
optG = torch.optim.AdamW([{'params': enc_p, 'lr': a.lr * 2}, {'params': unet_p, 'lr': a.lr if not a.full_ft else a.lr * 0.2}], weight_decay=1e-2)
optD = torch.optim.AdamW(Dn.parameters(), lr=a.lr, betas=(0.5, 0.999))
print('trainable M: encoder %.1f unet %.1f' % (sum(p.numel() for p in enc_p) / 1e6, sum(p.numel() for p in unet_p) / 1e6), flush=True)
cfg = vars(a).copy(); cfg['enc_params_M'] = sum(p.numel() for p in enc_p) / 1e6; cfg['clips'] = len(clips)
json.dump(cfg, open(a.out + '/cfg.json', 'w'))

SEQ = a.w_sync > 0
if SEQ:
    import syncloss
    syncloss.prep_mfcc(sorted({c['id'] for c in clips}), D.DATA, D.FR)
    snet = syncloss.load_net(dev)
ds = D.DS(clips, a.audio, seq=SEQ)
it = iter(torch.utils.data.DataLoader(ds, batch_size=a.bs // 5 if SEQ else a.bs, num_workers=12, drop_last=True, prefetch_factor=4))


def mouth_crops(imgs, face, size=96, out=128):
    """Crop a box around the lips (from the TRUE landmarks) from each image tensor in imgs."""
    c = face[:, LIPS, :2].mean(1) * imgs[0].shape[-1]
    res = [[] for _ in imgs]
    S = imgs[0].shape[-1]
    for i in range(c.shape[0]):
        x = int(min(max(c[i, 0].item() - size / 2, 0), S - size)); y = int(min(max(c[i, 1].item() - size / 2, 0), S - size))
        for k, im in enumerate(imgs): res[k].append(im[i:i + 1, :, y:y + size, x:x + size])
    return [F.interpolate(torch.cat(r), size=(out, out), mode='bilinear', align_corners=False) for r in res]


def t2im(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


t0 = time.time(); step = 0; log = open(a.out + '/log.txt', 'a'); acc = {}
while step < a.max_steps and time.time() - t0 < a.minutes * 60:
    b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
    if SEQ:
        mf_pos, mf_neg = b.pop('mf_pos'), b.pop('mf_neg')
        b = {k: v.flatten(0, 1) for k, v in b.items()}
    if a.no_audio: b['audio'] = torch.zeros_like(b['audio'])
    y = b['y']; B = y.shape[0]
    hide = torch.rand(B, device=dev) < a.p_hide
    with torch.autocast('cuda', dtype=torch.bfloat16):
        ref_lat = model.encode(b['ref'])
        noise = torch.randn(B, 4, 48, 48, device=dev)
        if SEQ: noise = noise[::5].repeat_interleave(5, 0); hide = hide[::5].repeat_interleave(5, 0)
        au_la = a.au_la if a.au_la >= 0 else random.choice([0, 2, 5, 10])
        lm_la = a.lm_la if a.lm_la >= 0 else random.choice([0, 0, 2, 5])
        if a.audio == 'win': b['audio'] = b['audio'][:, LAI[au_la]]
        o = model(b, noise, ref_lat, lm_la=lm_la, au_la=au_la, mask_lower=hide)
        out = o['img'].float()
        l2 = F.mse_loss(out, y)
        llp = lp(out, y).mean()
        face_t = b['face_w'][:, LM_PAST]
        mo, my = mouth_crops([out, y], face_t)
        lm = lp(mo, my).mean() + F.l1_loss(mo, my)
        hm = heatmap(face_t, b['pose_w'][:, LM_PAST], 96)
        hm_up = F.interpolate(hm, size=384, mode='bilinear', align_corners=False)
        lxy = torch.zeros((), device=dev)
        if hide.any() and a.enc != 'raster':
            h = o['hide']
            wl = torch.ones(h.shape[1], device=dev); wl[LIPS] = a.w_lips
            lxy = ((o['xy_pred'].float() - o['xy_true']).abs().sum(-1) * wl)[h].sum() / (wl.expand_as(h)[h].sum())
        loss = l2 + 5 * llp + a.w_mouth * lm + a.w_xy * lxy
        lsy = torch.zeros((), device=dev); dpos = lsy; dgap = lsy
        if SEQ:
            with torch.autocast('cuda', enabled=False):
                sin = out
                if a.sync_mouth:   # gradient may only reach the mouth box: everything else is the real frame
                    lp_ = face_t[:, LIPS, :2].float(); c_ = (lp_.min(1).values + lp_.max(1).values) / 2
                    io_ = (face_t[:, 263, :2] - face_t[:, 33, :2]).float().norm(dim=-1)
                    yy, xx = torch.meshgrid(torch.linspace(0, 1, 384, device=dev), torch.linspace(0, 1, 384, device=dev), indexing='ij')
                    mk = ((xx[None] - c_[:, 0, None, None]).abs() < 0.75 * io_[:, None, None]) & ((yy[None] - c_[:, 1, None, None]).abs() < 0.55 * io_[:, None, None])
                    mk = mk[:, None].float()
                    sin = out * mk + y * (1 - mk)
                lsy, dpos, dgap = syncloss.sync_loss(snet, sin, face_t, mf_pos, mf_neg)
            loss = loss + a.w_sync * lsy
        lg = torch.zeros((), device=dev)
        if step >= a.gan_start:
            lg = g_hinge(Dn, torch.cat([hm_up, out], 1))
            loss = loss + a.w_gan * lg
    optG.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
    optG.step()
    ld = torch.zeros(())
    if step >= a.gan_start - 200:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            ld = d_hinge(Dn, torch.cat([hm_up, y], 1), torch.cat([hm_up, out.detach()], 1))
        optD.zero_grad(set_to_none=True); ld.backward(); optD.step()
    for k, v in (('l2', l2), ('lpips', llp), ('mouth', lm), ('xy', lxy), ('g', lg), ('d', ld), ('sync', lsy), ('dpos', dpos), ('dgap', dgap)):
        acc[k] = acc.get(k, 0) + float(v)
    step += 1
    if step % 50 == 0:
        msg = f'step {step} t {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}G ' + ' '.join(f'{k} {v / 50:.4f}' for k, v in acc.items())
        print(msg, flush=True); log.write(msg + '\n'); log.flush(); acc = {}
    if step % 500 == 0 or step == 20:
        hv = (hm_up[:, :3].clamp(0, 1) * 2 - 1)
        grid = np.concatenate([np.concatenate(list(t2im(z)), 0) for z in (hv, b['ref'], out, y)], 1)
        cv2.imwrite(f'{a.out}/sample_{step:06d}.jpg', cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    if step % 1000 == 0:
        torch.save(model.trainable_state(), a.out + '/last.pt')
torch.save(model.trainable_state(), a.out + '/last.pt')
cfg.update(steps=step, train_minutes=(time.time() - t0) / 60)
json.dump(cfg, open(a.out + '/cfg.json', 'w'))
print('DONE', step, flush=True)
