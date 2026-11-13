"""SyncNet lip-sync loss (Wav2Lip-style 'expert'), mirroring the eval's preprocessing (eval/sync.py): 25 fps 5-frame windows,
face box from the mesh x1.106, crop [y-s, y+1.8s] x [x-1.4s, x+1.4s], pad 110, 224 px, BGR 0..255; MFCC 13x20 at 100 Hz."""
import os, sys, importlib.util, numpy as np, torch, torch.nn.functional as F
SD = os.environ.get('SYNCDIR', '/root/sync')
K_NEG = 6


def load_net(dev='cuda'):
    spec = importlib.util.spec_from_file_location('sn_model', f'{SD}/SyncNetModel.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    net = m.S(num_layers_in_fc_layers=1024)
    sd = torch.load(f'{SD}/syncnet_v2.model', map_location='cpu', weights_only=True); own = net.state_dict()
    for k, v in sd.items(): own[k].copy_(v)
    return net.to(dev).eval().requires_grad_(False)


def prep_mfcc(ids, data, fr):
    sys.path.insert(0, f'{SD}/pylib'); import python_speech_features, soundfile as sf
    for c in ids:
        p = f'{fr}/{c}/mfcc.npy'
        if os.path.exists(p) or not os.path.exists(f'{data}/clips/{c}/audio_pad.wav'): continue
        x, sr = sf.read(f'{data}/clips/{c}/audio_pad.wav', dtype='int16')
        if x.ndim > 1: x = x[:, 0]
        x = x[16000:16000 + 128000]
        np.save(p, np.stack([np.array(i) for i in zip(*python_speech_features.mfcc(x, sr))]).astype(np.float32))   # (13, ~799)


def crop224(img, face):
    """img (N,3,H,W) RGB in [-1,1]; face (N,478,3) xy01 of the SAME frames; N = W*5, box shared inside each 5-frame window."""
    N = img.shape[0]
    p = face[:, :468, :2].float()
    mn = p.min(1).values; mx = p.max(1).values
    c = (mn + mx) / 2; h = (mx - mn) / 2 * 1.106
    s = h.max(-1).values
    c = c.view(-1, 5, 2).median(1).values.repeat_interleave(5, 0); s = s.view(-1, 5).median(1).values.repeat_interleave(5, 0)
    # crop centre and half-size in [0,1] units: x in [cx-1.4s, cx+1.4s], y in [cy-s, cy+1.8s]
    cx = c[:, 0]; cy = c[:, 1] + 0.4 * s; hw = 1.4 * s
    theta = torch.zeros(N, 2, 3, device=img.device)
    theta[:, 0, 0] = hw * 2; theta[:, 0, 2] = cx * 2 - 1
    theta[:, 1, 1] = hw * 2; theta[:, 1, 2] = cy * 2 - 1
    x = (img.float() + 1) * 127.5 - 110.0
    g = F.affine_grid(theta, (N, 3, 224, 224), align_corners=False)
    out = F.grid_sample(x, g, mode='bilinear', padding_mode='zeros', align_corners=False) + 110.0
    return out[:, [2, 1, 0]]   # BGR


def sync_loss(net, img, face, mf_pos, mf_neg):
    """img (W*5,3,H,W); mf_pos (W,13,20); mf_neg (W,K,13,20). Returns (CE over [pos, negs] with logits = -distance, pos distance)."""
    Wn = mf_pos.shape[0]
    v = crop224(img, face).view(Wn, 5, 3, 224, 224).permute(0, 2, 1, 3, 4)
    fv = net.forward_lip(v)                                                    # W,1024
    mf = torch.cat([mf_pos[:, None], mf_neg], 1).reshape(-1, 1, 13, 20).float()
    fa = net.forward_aud(mf).view(Wn, -1, fv.shape[-1])                        # W,1+K,1024
    d = (fv[:, None] - fa).pow(2).sum(-1).add(1e-6).sqrt()                     # W,1+K
    ce = F.cross_entropy(-d, torch.zeros(Wn, dtype=torch.long, device=d.device))
    return ce, d[:, 0].mean(), (d[:, 1:].mean() - d[:, 0].mean())
