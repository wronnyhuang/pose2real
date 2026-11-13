"""Generate the heldout clips for a trained run at the run's size (videos are always written 512x640: lanczos-resized when the
run's size differs). Modes: full (all landmarks, default) and alips (lower-face landmarks hidden -> lips from audio).
--bench: batch 1, fp16, end to end (cond encoder -> UNet -> VAE decode -> uint8) ms/frame, then the same with LoRA fused +
torch.compile (PRIOR bench.py's 'compiled_ms'; --no_compile skips it) -> infer.json + bench.json."""
import argparse, os, json, time, subprocess
import torch, numpy as np
import data as D
from p2r import P2R

ap = argparse.ArgumentParser()
ap.add_argument('--run', required=True)          # folder with cfg.json + last.pt
ap.add_argument('--modes', default='full')       # full[,alips]
ap.add_argument('--splits', default='heldout')
ap.add_argument('--ema', type=float, default=0.0)  # blend of previous output latent
ap.add_argument('--au_la', type=int, default=-1)   # override lookahead at test time
ap.add_argument('--lm_la', type=int, default=-1)
ap.add_argument('--tag', default='')
ap.add_argument('--bs', type=int, default=24)
ap.add_argument('--bench', action='store_true')
ap.add_argument('--no_compile', action='store_true')
ap.add_argument('--clips', default='')           # comma list of clip ids (default: every clip of --splits)
ap.add_argument('--max_frames', type=int, default=0)
ap.add_argument('--dev', default='cuda')
a = ap.parse_args()
cfg = json.load(open(a.run + '/cfg.json'))
dev = a.dev; dt = dev.split(':')[0]
W, H = map(int, cfg.get('size', '512x640').lower().split('x'))
OW, OH = 512, 640
adim = 512 if cfg['audio'] == 'loc' else 768
model = P2R(cfg['backbone'], cfg['inj'], cfg['enc'], cfg['depth'], adim, cfg['rank'], None, full_ft=cfg.get('full_ft', False),
            refbase=cfg.get('refbase', False), hands=cfg.get('hands', 'tokens'), size=(W, H)).to(dev)
model.load_trainable(torch.load(a.run + '/last.pt', map_location='cpu'))
fp = torch.float16 if dt == 'cuda' else (torch.bfloat16 if os.environ.get('CPU_AMP', 'bf16') == 'bf16' else torch.float32)   # CPU smoke test
model.eval().to(fp)
au_la = (cfg['au_la'] if cfg['au_la'] >= 0 else 5) if a.au_la < 0 else a.au_la
lm_la = (cfg['lm_la'] if cfg['lm_la'] >= 0 else 0) if a.lm_la < 0 else a.lm_la
clips = D.ready_ids(a.splits.split(','), cfg['audio'])
if a.clips: clips = [c for c in clips if c['id'] in a.clips.split(',')]
allc = D.ready_ids(['heldout_ref', 'heldout'], cfg['audio'])
LAI = {0: 0, 2: 1, 5: 2, 10: 3}
gpu = torch.cuda.get_device_name() if dt == 'cuda' else 'cpu'


def ref_for(c):
    """Reference image: middle frame of another window of the same person (heldout_ref), else frame 0 of the clip itself."""
    for x in allc:
        if x['split'] == 'heldout_ref' and x['identity'] == c['identity']:
            return f"{D.FR}/{x['id']}/0120.jpg", 'heldout_ref', (x['id'], 120)
    return f"{D.FR}/{c['id']}/0000.jpg", 'own_frame0', (c['id'], 0)


def t2im(x):
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()


class VW:
    """Raw RGB WxH frames -> H.264 yuv420p 30 fps at 512x640 (lanczos when the run's size differs)."""

    def __init__(self, path, fps=30):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        vf = [] if (W, H) == (OW, OH) else ['-vf', f'scale={OW}:{OH}:flags=lanczos']
        self.p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps),
                                   '-i', '-'] + vf + ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'fast', path],
                                  stdin=subprocess.PIPE)

    def write(self, im): self.p.stdin.write(np.ascontiguousarray(im).tobytes())

    def close(self): self.p.stdin.close(); assert self.p.wait() == 0, 'ffmpeg failed'


def batch(cl, ts, rf, rpz):
    w = [cl.window(t) for t in ts]
    b = {k: torch.from_numpy(np.stack([x[i] for x in w])).to(dev, fp) for i, k in enumerate(('face_w', 'pose_w', 'hands_w', 'audio'))}
    b['ref_face'] = rf.expand(len(ts), -1, -1); b['ref_pose'] = rpz.expand(len(ts), -1, -1)
    if cfg['audio'] == 'win': b['audio'] = b['audio'][:, LAI[au_la]]
    if cfg.get('no_audio'): b['audio'] = torch.zeros_like(b['audio'])
    return b


def setup_ref(c):
    rp, rsrc, (rid, rfr) = ref_for(c)
    rcl = D.Clip(rid, cfg['audio'])
    rf = torch.from_numpy(rcl.face[rfr])[None].to(dev, fp); rpz = torch.from_numpy(rcl.pose[rfr])[None].to(dev, fp)
    ref_lat = model.encode(D.to_t(D.rd(rp, (W, H))).unsqueeze(0).to(dev, fp))
    g = torch.Generator(device=dev).manual_seed(1234)
    noise = torch.randn(1, 4, H // 8, W // 8, device=dev, generator=g).to(fp)     # ONE fixed noise per clip -> no per-frame flicker
    return rsrc, rf, rpz, ref_lat, noise


info = {}; nfr = 0
with torch.no_grad(), torch.autocast(dt, dtype=fp, enabled=fp != torch.float32):
    for mode in a.modes.split(','):
        outd = f"{a.run}/out{'' if mode == 'full' else '_' + mode}{a.tag}"
        for c in clips:
            cl = D.Clip(c['id'], cfg['audio'])
            rsrc, rf, rpz, ref_lat, noise = setup_ref(c); info[c['id']] = rsrc
            vw = VW(f"{outd}/{c['id']}.mp4"); prev = None
            T = min(cl.T, a.max_frames) if a.max_frames else cl.T
            for s in range(0, T, a.bs):
                ts = list(range(s, min(s + a.bs, T))); n = len(ts)
                b = batch(cl, ts, rf, rpz)
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
                assert img.shape[-2:] == (H, W), img.shape
                for im in t2im(img): vw.write(im); nfr += 1
            vw.close()
        print('wrote', outd, len(clips), 'clips', flush=True)
bench = {}
if a.bench:  # batch 1, end to end: conditioning encoder + UNet + VAE decode -> uint8
    c = clips[0]; cl = D.Clip(c['id'], cfg['audio'])
    with torch.no_grad(), torch.autocast(dt, dtype=fp, enabled=fp != torch.float32):
        rsrc, rf, rpz, ref_lat, noise = setup_ref(c)
        b = batch(cl, [min(100, cl.T - 1)], rf, rpz)
        hide = torch.zeros(1, dtype=torch.bool, device=dev)
        sync = torch.cuda.synchronize if dt == 'cuda' else (lambda: None)

        def full():
            o = model(b, noise, ref_lat, lm_la=lm_la, au_la=au_la, mask_lower=hide); return t2im(o['img'])

        def T_(f, n=50, warm=10):
            if dt != 'cuda': n, warm = 2, 1
            for _ in range(warm): f()
            sync(); t0 = time.time()
            for _ in range(n): f()
            sync(); return round((time.time() - t0) / n * 1000, 2)

        bench['plain_ms'] = T_(full)
        bench['cond_ms'] = T_(lambda: model.cond(b, lm_la=lm_la, au_la=au_la, mask_lower=hide))
        x = model(b, noise, ref_lat, lm_la=lm_la, au_la=au_la, mask_lower=hide, decode=False)['x0']
        bench['vae_ms'] = T_(lambda: model.vae.decode(x / model.sf).sample)
        print('ms/frame', bench['plain_ms'], gpu, flush=True)
        if not a.no_compile and dt == 'cuda':
            try:
                if not cfg.get('full_ft'): model.unet.fuse_lora()
                model.unet = torch.compile(model.unet, mode='max-autotune-no-cudagraphs')
                model.vae.decoder = torch.compile(model.vae.decoder, mode='max-autotune-no-cudagraphs')
                for _ in range(4): full()
                bench['compiled_ms'] = T_(full)
                print('compiled ms/frame', bench['compiled_ms'], flush=True)
            except Exception as e:
                bench['compile_error'] = str(e)[:300]; print('compile failed', bench['compile_error'], flush=True)
    bench.update(gpu=gpu, size=f'{W}x{H}')
    json.dump(bench, open(a.run + '/bench.json', 'w'))
    print('BENCH', json.dumps(bench), flush=True)
cfg.update(ref_source=info, ms_per_frame=bench.get('plain_ms'), compiled_ms=bench.get('compiled_ms'), gpu=gpu, test_au_la=au_la,
           test_lm_la=lm_la, ema=a.ema, frames_written=nfr, out_size='512x640')
json.dump(cfg, open(f"{a.run}/infer{a.tag}.json", 'w'), indent=1)
