"""Generate the heldout clips for a trained run. Modes: full (all landmarks) and alips (lower-face landmarks hidden -> lips from audio)."""
import argparse, os, json, time, subprocess
import torch, numpy as np
import data as D
from p2r import P2R

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)          # folder with cfg.json + last.pt
ap.add_argument('--modes', default='full,alips')
ap.add_argument('--splits', default='heldout')
ap.add_argument('--ema', type=float, default=0.0)  # blend of previous output latent
ap.add_argument('--au_la', type=int, default=-1)   # override lookahead at test time
ap.add_argument('--lm_la', type=int, default=-1)
ap.add_argument('--tag', default='')
ap.add_argument('--bs', type=int, default=24)
ap.add_argument('--bench', action='store_true')
a = ap.parse_args()
cfg = json.load(open(a.run + '/cfg.json'))
dev = 'cuda'
adim = 512 if cfg['audio'] == 'loc' else 768
model = P2R(cfg['backbone'], cfg['inj'], cfg['enc'], cfg['depth'], adim, cfg['rank'], None, full_ft=cfg.get('full_ft', False), refbase=cfg.get('refbase', False)).to(dev)
model.load_trainable(torch.load(a.run + '/last.pt'))
model.eval().half()
au_la = (cfg['au_la'] if cfg['au_la'] >= 0 else 5) if a.au_la < 0 else a.au_la
lm_la = (cfg['lm_la'] if cfg['lm_la'] >= 0 else 0) if a.lm_la < 0 else a.lm_la
clips = D.ready_ids(a.splits.split(','), cfg['audio'])
allc = D.ready_ids(['heldout_ref', 'heldout'])
LAI = {0: 0, 2: 1, 5: 2, 10: 3}


def ref_for(c):
    """Reference image: middle frame of another window of the same person (heldout_ref), else frame 0 of the clip itself."""
    for x in allc:
        if x['split'] == 'heldout_ref' and x['identity'] == c['identity']:
            return f"{D.FR}/{x['id']}/0120.jpg", 'heldout_ref', (x['id'], 120)
    return f"{D.FR}/{c['id']}/0000.jpg", 'own_frame0', (c['id'], 0)


def t2im(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


class VW:
    def __init__(self, path, fps=30, size=384):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{size}x{size}', '-r', str(fps),
                                   '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'fast', path], stdin=subprocess.PIPE)

    def write(self, im): self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self): self.p.stdin.close(); self.p.wait()


info = {}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    for mode in a.modes.split(','):
        name = os.path.basename(a.run.rstrip('/')) + ('' if mode == 'full' else '__' + mode) + a.tag
        outd = f"{a.run}/out{'' if mode == 'full' else '_' + mode}{a.tag}"
        for c in clips:
            cl = D.Clip(c['id'], cfg['audio'])
            rp, rsrc, (rid, rfr) = ref_for(c); info[c['id']] = rsrc
            rcl = D.Clip(rid, cfg['audio'])
            rf = torch.from_numpy(rcl.face[rfr])[None].to(dev).half(); rpz = torch.from_numpy(rcl.pose[rfr])[None].to(dev).half()
            ref = D.to_t(D.rd(rp)).unsqueeze(0).to(dev).half()
            ref_lat = model.encode(ref)
            g = torch.Generator(device=dev).manual_seed(1234)
            noise = torch.randn(1, 4, 48, 48, device=dev, generator=g).half()     # ONE fixed noise per clip -> no per-frame flicker
            vw = VW(f"{outd}/{c['id']}.mp4"); prev = None
            for s in range(0, cl.T, a.bs):
                ts = list(range(s, min(s + a.bs, cl.T)))
                w = [cl.window(t) for t in ts]
                b = {'face_w': torch.from_numpy(np.stack([x[0] for x in w])).to(dev).half(),
                     'pose_w': torch.from_numpy(np.stack([x[1] for x in w])).to(dev).half(),
                     'audio': torch.from_numpy(np.stack([x[2] for x in w])).to(dev).half()}
                b['ref_face'] = rf.expand(len(ts), -1, -1); b['ref_pose'] = rpz.expand(len(ts), -1, -1)
                if cfg['audio'] == 'win': b['audio'] = b['audio'][:, LAI[au_la]]
                if cfg.get('no_audio'): b['audio'] = torch.zeros_like(b['audio'])
                n = len(ts)
                hide = torch.full((n,), mode == 'alips', dtype=torch.bool, device=dev)
                o = model(b, noise.expand(n, -1, -1, -1), ref_lat.expand(n, -1, -1, -1), lm_la=lm_la, au_la=au_la, mask_lower=hide, decode=a.ema <= 0)
                if a.ema > 0:
                    xs = []
                    for k in range(n):
                        x = o['x0'][k:k + 1]
                        prev = x if prev is None else a.ema * prev + (1 - a.ema) * x
                        xs.append(prev)
                    img = model.vae.decode(torch.cat(xs) / model.sf).sample
                else:
                    img = o['img']
                for im in t2im(img): vw.write(im)
            vw.close()
        print('wrote', outd, flush=True)
    ms = None
    if a.bench:  # batch 1, fp16, end to end: conditioning encoder + UNet + VAE decode -> uint8
        cl = D.Clip(clips[0]['id'], cfg['audio']); w = cl.window(100)
        b = {'face_w': torch.from_numpy(w[0])[None].to(dev).half(), 'pose_w': torch.from_numpy(w[1])[None].to(dev).half(),
             'audio': torch.from_numpy(w[2])[None].to(dev).half()}
        b['ref_face'] = rf; b['ref_pose'] = rpz
        if cfg['audio'] == 'win': b['audio'] = b['audio'][:, LAI[au_la]]
        hide = torch.zeros(1, dtype=torch.bool, device=dev)
        for i in range(60):
            if i == 10: torch.cuda.synchronize(); t0 = time.time()
            o = model(b, noise, ref_lat, lm_la=lm_la, au_la=au_la, mask_lower=hide); _ = t2im(o['img'])
        torch.cuda.synchronize(); ms = (time.time() - t0) / 50 * 1000
        print('ms/frame', ms, torch.cuda.get_device_name(), flush=True)
cfg.update(ref_source=info, ms_per_frame=ms, gpu=torch.cuda.get_device_name(), test_au_la=au_la, test_lm_la=lm_la, ema=a.ema)
json.dump(cfg, open(f"{a.run}/infer{a.tag}.json", 'w'), indent=1)
