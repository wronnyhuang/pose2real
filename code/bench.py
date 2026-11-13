"""Speed of a trained run, batch 1, fp16, end to end (conditioning encoder -> UNet -> VAE decode -> uint8), with LoRA fused + torch.compile."""
import sys, json, time, torch, numpy as np
import data as D
from p2r import P2R
run = sys.argv[1]
cfg = json.load(open(run + '/cfg.json'))
adim = 512 if cfg['audio'] == 'loc' else 768
m = P2R(cfg['backbone'], cfg['inj'], cfg['enc'], cfg['depth'], adim, cfg['rank'], None, refbase=cfg.get('refbase', False)).cuda()
m.load_trainable(torch.load(run + '/last.pt')); m.eval().half()
m.unet.fuse_lora()
c = D.ready_ids(['heldout'], cfg['audio'])[0]; cl = D.Clip(c['id'], cfg['audio']); w = cl.window(100)
au = torch.from_numpy(w[2])[None].cuda().half()
if cfg['audio'] == 'win': au = au[:, 2]
b = {'face_w': torch.from_numpy(w[0])[None].cuda().half(), 'pose_w': torch.from_numpy(w[1])[None].cuda().half(), 'audio': au,
     'ref_face': torch.from_numpy(cl.face[0])[None].cuda().half(), 'ref_pose': torch.from_numpy(cl.pose[0])[None].cuda().half()}
noise = torch.randn(1, 4, 48, 48).cuda().half(); ref = torch.randn(1, 4, 48, 48).cuda().half()
hide = torch.zeros(1, dtype=torch.bool).cuda()


def T(f, n=60):
    for _ in range(8): f()
    torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.time() - t) / n * 1000, 2)


def full():
    o = m(b, noise, ref, mask_lower=hide)
    return ((o['img'].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu()


res = {'gpu': torch.cuda.get_device_name()}
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
    res['plain_ms'] = T(full)
    res['cond_ms'] = T(lambda: m.cond(b, mask_lower=hide))
    x = m(b, noise, ref, decode=False)['x0']
    res['vae_ms'] = T(lambda: m.vae.decode(x / m.sf).sample)
    try:
        m.unet = torch.compile(m.unet, mode='max-autotune-no-cudagraphs'); m.vae.decoder = torch.compile(m.vae.decoder, mode='max-autotune-no-cudagraphs')
        for _ in range(4): full()
        res['compiled_ms'] = T(full)
    except Exception as e:
        res['compile_error'] = str(e)[:200]
print('BENCH', json.dumps(res))
json.dump(res, open(run + '/bench.json', 'w'))
