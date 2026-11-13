"""Standalone trainer for the conditioning encoder's audio -> lower-face landmark prediction (no diffusion model).
Fast (minutes): used to sweep audio features / lookahead / depth, and to warm-start the full model's encoder."""
import argparse, os, time, json, random
import torch, numpy as np
import data as D
from p2r import CondEncoder, LIPS, LM_PAST, NF

ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
ap.add_argument('--audio', default='win')
ap.add_argument('--au_la', type=int, default=-1)
ap.add_argument('--lm_la', type=int, default=0)
ap.add_argument('--depth', type=int, default=6)
ap.add_argument('--refbase', type=int, default=1)
ap.add_argument('--w_lips', type=float, default=3.0)
ap.add_argument('--no_audio', action='store_true')
ap.add_argument('--origin', default='all')
ap.add_argument('--steps', type=int, default=6000)
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--lr', type=float, default=3e-4)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'; LAI = {0: 0, 2: 1, 5: 2, 10: 3}
clips = D.ready_ids(['train'], a.audio)
if a.origin != 'all': clips = [c for c in clips if c.get('origin', 'render2real') == a.origin]
held = D.ready_ids(['heldout'], a.audio); refs = D.ready_ids(['heldout_ref'], a.audio)
low = D.lower_face_idx(clips)
adim = 512 if a.audio == 'loc' else 768
enc = CondEncoder(depth=a.depth, audio_dim=adim, inj='noise', lower_idx=low, refbase=bool(a.refbase)).to(dev)
opt = torch.optim.AdamW(enc.parameters(), lr=a.lr, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
it = iter(torch.utils.data.DataLoader(D.DS(clips, a.audio, noimg=True), batch_size=a.bs, num_workers=8, drop_last=True))
print('clips', len(clips), 'lower pts', int(low.sum()), flush=True)


def run(b, au_la, hide):
    if a.audio == 'win': b = dict(b, audio=b['audio'][:, LAI[au_la]])
    if a.no_audio: b = dict(b, audio=torch.zeros_like(b['audio']))
    return enc(b, lm_la=a.lm_la, au_la=au_la, mask_lower=hide, xy_only=True)


@torch.no_grad()
def evaluate(au_la):
    enc.eval(); res = []
    for c in held:
        cl = D.Clip(c['id'], a.audio)
        r = next((x for x in refs if x['identity'] == c['identity']), None)
        rcl = D.Clip(r['id'], a.audio) if r else cl; rfr = 120 if r else 0
        w = [cl.window(t) for t in range(cl.T)]
        b = {'face_w': torch.from_numpy(np.stack([x[0] for x in w])).to(dev), 'pose_w': torch.from_numpy(np.stack([x[1] for x in w])).to(dev),
             'audio': torch.from_numpy(np.stack([x[2] for x in w])).to(dev)}
        b['ref_face'] = torch.from_numpy(rcl.face[rfr])[None].to(dev).expand(cl.T, -1, -1); b['ref_pose'] = torch.from_numpy(rcl.pose[rfr])[None].to(dev).expand(cl.T, -1, -1)
        ps, tt = [], []
        for i in range(0, cl.T, 48):
            o = run({k: v[i:i + 48] for k, v in b.items()}, au_la, torch.ones(len(b['face_w'][i:i + 48]), dtype=torch.bool, device=dev))
            ps.append(o['xy_pred'].float()); tt.append(o['xy_true'])
        p = torch.cat(ps); t = torch.cat(tt)
        io = (t[:, 263] - t[:, 33]).norm(dim=-1)
        nme = ((p[:, LIPS] - t[:, LIPS]).norm(dim=-1).mean(1) / io).mean().item() * 100
        gp = (p[:, 13] - p[:, 14]).norm(dim=-1).cpu().numpy(); gt = (t[:, 13] - t[:, 14]).norm(dim=-1).cpu().numpy()
        wp = (p[:, 78] - p[:, 308]).norm(dim=-1).cpu().numpy(); wt = (t[:, 78] - t[:, 308]).norm(dim=-1).cpu().numpy()
        res.append((nme, np.corrcoef(gp, gt)[0, 1], np.corrcoef(wp, wt)[0, 1]))
    enc.train()
    r = np.nanmean(np.array(res), 0)
    return {'mouth_nme': round(float(r[0]), 3), 'open_corr': round(float(r[1]), 3), 'width_corr': round(float(r[2]), 3)}


t0 = time.time(); acc = 0
for step in range(1, a.steps + 1):
    b = {k: v.to(dev, non_blocking=True) for k, v in next(it).items()}
    B = b['face_w'].shape[0]
    au_la = a.au_la if a.au_la >= 0 else random.choice([0, 2, 5, 10])
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = run(b, au_la, torch.ones(B, dtype=torch.bool, device=dev))
    h = o['hide']; wl = torch.ones(h.shape[1], device=dev); wl[LIPS] = a.w_lips
    loss = ((o['xy_pred'].float() - o['xy_true']).abs().sum(-1) * wl)[h].sum() / wl.expand_as(h)[h].sum()
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0); opt.step(); sched.step()
    acc += loss.item()
    if step % 500 == 0:
        las = [a.au_la] if a.au_la >= 0 else [0, 5]
        print(f'step {step} t {time.time() - t0:.0f}s xy {acc / 500:.4f}', {l: evaluate(l) for l in las}, flush=True); acc = 0
las = [a.au_la] if a.au_la >= 0 else [0, 2, 5, 10]
final = {str(l): evaluate(l) for l in las}
print('FINAL', json.dumps(final), flush=True)
torch.save({k: v.cpu() for k, v in enc.state_dict().items()}, a.out + '/enc.pt')
json.dump(dict(vars(a), final=final, clips=len(clips), seconds=time.time() - t0), open(a.out + '/xy.json', 'w'), indent=1)
