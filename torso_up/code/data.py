"""Pod-side data: frame extraction, audio features, dataset (512x640 torso-up frames, face + pose + hands)."""
import os, json, random, subprocess
import numpy as np, cv2, torch
from p2r import LM_PAST, LM_FUT, AU_PAST, AU_FUT, NF, NP

DATA = os.environ.get('DATA', '/root/data')
FR = os.environ.get('FRAMES', '/root/frames')
N = 240
SRC = (512, 640)    # (W, H) of the frames on disk


def manifest():
    return json.load(open(f'{DATA}/manifest.json'))['clips']


def _save(p, x):   # atomic: a killed prep never leaves a truncated feature file behind
    np.save(p + '.tmp.npy', x); os.replace(p + '.tmp.npy', p)


def prep_frames(ids, jobs_max=16):
    jobs = []
    for cid in ids:
        d = f'{FR}/{cid}'
        if os.path.exists(f'{d}/ok') or not os.path.exists(f'{DATA}/clips/{cid}/real.mp4'): continue
        os.makedirs(d, exist_ok=True)
        jobs.append(subprocess.Popen(f"ffmpeg -v error -y -i {DATA}/clips/{cid}/real.mp4 -q:v 2 -start_number 0 {d}/%04d.jpg && touch {d}/ok", shell=True))
        if len(jobs) >= jobs_max: jobs.pop(0).wait()
    for j in jobs: j.wait()


def _wav(c):
    import soundfile as sf
    x, sr = sf.read(f'{DATA}/clips/{c}/audio_pad.wav', dtype='float32'); assert sr == 16000, sr
    return x.mean(1) if x.ndim > 1 else x


@torch.no_grad()
def prep_audio(ids, dev='cuda'):
    """HuBERT-base: local conv features (512-d, strictly local 25 ms receptive field -> honest lookahead control) and
    layer-9 contextual features (768-d, sees the whole 10 s clip -> 'unlimited lookahead' upper bound). 50 Hz, float16."""
    from transformers import HubertModel
    todo = [c for c in ids if not os.path.exists(f'{FR}/{c}/au_ctx.npy') and os.path.exists(f'{DATA}/clips/{c}/audio_pad.wav')]
    if not todo: return
    m = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(dev).eval()
    for i in range(0, len(todo), 16):
        ch = todo[i:i + 16]
        w = []
        for c in ch:
            x = _wav(c)
            x = np.pad(x, (0, max(0, 160000 - len(x))))[:160000]
            w.append((x - x.mean()) / (x.std() + 1e-7))
        w = torch.from_numpy(np.stack(w)).to(dev)
        o = m(w, output_hidden_states=True)
        loc = m.feature_extractor(w).transpose(1, 2)            # B,499,512 local conv features
        loc = torch.nn.functional.layer_norm(loc, (512,))
        ctx = o.hidden_states[9]
        for k, c in enumerate(ch):
            os.makedirs(f'{FR}/{c}', exist_ok=True)
            _save(f'{FR}/{c}/au_loc.npy', loc[k].cpu().numpy().astype(np.float16))
            _save(f'{FR}/{c}/au_ctx.npy', ctx[k].cpu().numpy().astype(np.float16))
        print(f'audio loc/ctx {i + len(ch)}/{len(todo)}', flush=True)
    del m


WIN_LA = [0, 2, 5, 10]     # lookahead settings (video frames) for the causal windowed HuBERT features
WIN_TOK = 24


@torch.no_grad()
def prep_audio_win(ids, dev='cuda'):
    """Honestly causal contextual features: for video frame t and lookahead L, HuBERT-base runs on the 2.0 s of audio that ENDS
    at (end of frame t + L/30 s) and nothing later; the last 24 layer-9 tokens are kept. -> au_win.npy (240,4,24,768) f16."""
    from transformers import HubertModel
    todo = [c for c in ids if not os.path.exists(f'{FR}/{c}/au_win.npy') and os.path.exists(f'{DATA}/clips/{c}/audio_pad.wav')]
    if not todo: return
    half = str(dev).startswith('cuda')
    m = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(dev).eval()
    if half: m = m.half()
    for n, c in enumerate(todo):
        x = _wav(c)
        x = np.pad(x, (32000, max(0, 160000 - len(x)) + 16000))      # 2 s of zeros in front so every window is full length
        out = np.zeros((N, len(WIN_LA), WIN_TOK, 768), np.float16)
        for li, L in enumerate(WIN_LA):
            ends = [int(round(((t + 1 + L) / 30 + 1.0) * 16000)) + 32000 for t in range(N)]
            w = np.stack([x[e - 32000:e] for e in ends])
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-5)
            for s in range(0, N, 120 if half else 40):
                wt = torch.from_numpy(w[s:s + (120 if half else 40)]).to(dev)
                h = m(wt.half() if half else wt, output_hidden_states=True).hidden_states[9]
                out[s:s + len(wt), li] = h[:, -WIN_TOK:].float().cpu().numpy()
        os.makedirs(f'{FR}/{c}', exist_ok=True)
        _save(f'{FR}/{c}/au_win.npy', out)
        if n % 20 == 0: print(f'audio win {n + 1}/{len(todo)}', flush=True)
    del m


def ready_ids(splits, audio='ctx', sets=None):
    out = []
    for c in manifest():
        if c['split'] not in splits or (sets and c.get('set') not in sets): continue
        if all(os.path.exists(p) for p in (f"{FR}/{c['id']}/ok", f"{FR}/{c['id']}/au_{audio}.npy", f"{DATA}/clips/{c['id']}/lm.npz")):
            out.append(c)
    return out


class Clip:
    def __init__(self, cid, audio='loc'):
        z = np.load(f'{DATA}/clips/{cid}/lm.npz')
        self.id = cid
        face = z['face'].astype(np.float32); pose = z['pose'].astype(np.float32)
        fok = z['face_ok'].astype(bool); pok = z['pose_ok'].astype(bool)
        T = len(face)
        self.fok = fok.copy()
        # face / pose: hold last good value through gaps
        for arr, ok in ((face, fok), (pose, pok)):
            last = None
            first = np.argmax(ok) if ok.any() else 0
            for t in range(len(arr)):
                if ok[t]: last = arr[t]
                elif last is not None: arr[t] = last
                else: arr[t] = arr[first]
        if not pok.any(): pose[:] = 0
        # hands: NOT held through gaps -- a missing hand is simply not visible. (T,2,21,4) = x, y, z, ok
        if 'hands' in z.files:
            hk = z['hands_ok'].astype(bool)[:T]; hd = z['hands'].astype(np.float32)[:T]
        else:
            hk = np.zeros((T, 2), bool); hd = np.zeros((T, 2, 21, 3), np.float32)
        hd = hd * hk[:, :, None, None]
        self.hands = np.concatenate([hd, np.broadcast_to(hk[:, :, None, None], (T, 2, 21, 1)).astype(np.float32)], -1)
        self.face, self.pose = face, pose
        self.au = np.load(f'{FR}/{cid}/au_{audio}.npy', mmap_mode='r' if audio == 'win' else None)
        self.win = audio == 'win'
        self.T = min(N, T)
        self.hand_t = [t for t in range(self.T) if hk[t].any() and fok[t]]   # frames with >= 1 visible hand

    def window(self, t):
        """-> face (W,478,3), pose (W,33,4), hands (W,2,21,4), audio."""
        idx = np.clip(np.arange(t - LM_PAST, t + LM_FUT + 1), 0, self.T - 1)
        lm = (self.face[idx].copy(), self.pose[idx].copy(), self.hands[idx].copy())
        if self.win: return lm + (np.asarray(self.au[t]).astype(np.float32),)
        k = int(round(((t + 0.5) / 30 + 1.0) * 50 - 0.5))
        ai = np.arange(k - AU_PAST, k + AU_FUT + 1)
        au = self.au[np.clip(ai, 0, len(self.au) - 1)].astype(np.float32)
        au[(ai < 0) | (ai >= len(self.au))] = 0
        return lm + (au,)


def rd(p, size=SRC):
    """Read a frame as RGB at size (W, H) (area-resized when it differs from the stored 512x640)."""
    im = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    if (im.shape[1], im.shape[0]) != tuple(size): im = cv2.resize(im, tuple(size), interpolation=cv2.INTER_AREA)
    return im


def to_t(im):
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(127.5).sub(1)


def rand_aff(W, H):
    """Small random similarity about the frame centre, in pixels. No flip (handedness)."""
    s = random.uniform(0.94, 1.08); ang = random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
    M[0, 2] += random.uniform(-0.04, 0.04) * W; M[1, 2] += random.uniform(-0.04, 0.04) * H
    return M


def aff_xy(xy01, M, W, H):
    """Apply the pixel-space affine M to per-axis normalised xy."""
    S = np.array([W, H], np.float32)
    return ((xy01 * S) @ M[:, :2].T.astype(np.float32) + M[:, 2].astype(np.float32)) / S


def aug_lm(fw, pw, hw, M, W, H):
    fw[..., :2] = aff_xy(fw[..., :2], M, W, H); pw[..., :2] = aff_xy(pw[..., :2], M, W, H)
    hw[..., :2] = aff_xy(hw[..., :2], M, W, H) * hw[..., 3:4]       # missing hands stay at 0


class DS(torch.utils.data.Dataset):
    """size = (W, H) of the training images. p_hand: probability that a sample is drawn from a frame with >= 1 visible hand."""

    def __init__(self, clips, audio='loc', length=10 ** 7, aug=True, noimg=False, size=SRC, p_hand=0.0):
        self.meta = clips; self.audio = audio; self.length = length; self.aug = aug; self.noimg = noimg
        self.W, self.H = size; self.p_hand = p_hand
        self.by_id = {}
        for c in clips: self.by_id.setdefault(c['identity'], []).append(c['id'])
        self.idn = {c['id']: c['identity'] for c in clips}
        self.hmeta = [c for c in clips if c.get('hand_frac', 1) > 0]   # candidates; re-checked on the loaded landmarks
        self.cache = {}

    def clip(self, cid):
        if cid not in self.cache: self.cache[cid] = Clip(cid, self.audio)
        return self.cache[cid]

    def __len__(self): return self.length

    def pick(self):
        if self.hmeta and random.random() < self.p_hand:
            for _ in range(20):
                c = self.clip(random.choice(self.hmeta)['id'])
                if c.hand_t: return c, random.choice(c.hand_t)
        while True:
            c = self.clip(random.choice(self.meta)['id'])
            t = random.randrange(0, c.T)
            if c.fok[t]: return c, t

    def ref_of(self, c, t):
        """Identity reference: another clip of the same person when there is one, else a frame >= 2 s away."""
        others = [x for x in self.by_id[self.idn[c.id]] if x != c.id]
        if others and random.random() < 0.7: rc = self.clip(random.choice(others)); rt = random.randrange(0, rc.T)
        else: rc = c; rt = (t + random.randrange(60, N - 60)) % c.T
        return rc, rt

    def __getitem__(self, i):
        c, t = self.pick()
        rcl, rt = self.ref_of(c, t)
        fw, pw, hw, au = c.window(t)
        W, H = self.W, self.H
        out = {'ref_face': torch.from_numpy(rcl.face[rt].copy()), 'ref_pose': torch.from_numpy(rcl.pose[rt].copy()), 'audio': torch.from_numpy(au)}
        if self.noimg:
            if self.aug: aug_lm(fw, pw, hw, rand_aff(W, H), W, H)
            return dict(out, face_w=torch.from_numpy(fw), pose_w=torch.from_numpy(pw), hands_w=torch.from_numpy(hw))
        y = rd(f'{FR}/{c.id}/{t:04d}.jpg', (W, H))
        rp = f'{FR}/{rcl.id}/{rt:04d}.jpg'
        ref = rd(rp, (W, H)) if os.path.exists(rp) else y
        if self.aug:
            M = rand_aff(W, H)
            y = cv2.warpAffine(y, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            aug_lm(fw, pw, hw, M, W, H)
            ref = cv2.warpAffine(ref, rand_aff(W, H), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.5:
                ref = np.clip((ref.astype(np.float32) - 128) * random.uniform(0.92, 1.08) + 128 + random.uniform(-8, 8), 0, 255).astype(np.uint8)
        return dict(out, y=to_t(y), ref=to_t(ref), face_w=torch.from_numpy(fw), pose_w=torch.from_numpy(pw), hands_w=torch.from_numpy(hw))


def lower_face_idx(clips, n=40):
    """Points of the lower face (below the nose base, pt 2) in the mean face-normalised shape: the set hidden in 'audio-lips' mode.
    Measured in pixels of the 512x640 frame (x and y are normalised per axis)."""
    acc = []
    S = np.array(SRC, np.float64)
    for c in clips[:n]:
        z = np.load(f"{DATA}/clips/{c['id']}/lm.npz"); ok = z['face_ok'].astype(bool)
        if ok.sum() < 10: continue
        f = z['face'][ok][::10, :, :2].astype(np.float64) * S
        # normalise by the eye line: rotate so outer eye corners (33, 263) are horizontal
        e = f[:, 263] - f[:, 33]; ang = np.arctan2(e[:, 1], e[:, 0])
        cs, sn = np.cos(-ang), np.sin(-ang)
        g = f - f[:, 2:3]
        yy = g[..., 0] * sn[:, None] + g[..., 1] * cs[:, None]
        acc.append((yy / np.linalg.norm(e, axis=1)[:, None]).mean(0))
    yy = np.mean(acc, 0)
    low = np.zeros(NF + NP, bool); low[:NF] = yy > 0.02
    low[NF + 9] = low[NF + 10] = True   # pose mouth corners
    return torch.from_numpy(low)
