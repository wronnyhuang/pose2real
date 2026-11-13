"""Pod-side data: frame extraction, audio features, dataset."""
import os, json, random, subprocess, sys, glob
import numpy as np, cv2, torch
from p2r import LM_PAST, LM_FUT, AU_PAST, AU_FUT, NF, NP

DATA = os.environ.get('DATA', '/root/data')
FR = os.environ.get('FRAMES', '/root/frames')
N = 240


def manifest():
    return json.load(open(f'{DATA}/manifest.json'))['clips']


def prep_frames(ids):
    jobs = []
    for cid in ids:
        d = f'{FR}/{cid}'
        if os.path.exists(f'{d}/ok') or not os.path.exists(f'{DATA}/clips/{cid}/real.mp4'): continue
        os.makedirs(d, exist_ok=True)
        jobs.append(subprocess.Popen(f"ffmpeg -v error -y -i {DATA}/clips/{cid}/real.mp4 -q:v 2 -start_number 0 {d}/%04d.jpg && touch {d}/ok", shell=True))
        if len(jobs) >= 16: jobs.pop(0).wait()
    for j in jobs: j.wait()


@torch.no_grad()
def prep_audio(ids, dev='cuda'):
    """HuBERT-base: local conv features (512-d, strictly local 25 ms receptive field -> honest lookahead control) and
    layer-9 contextual features (768-d, sees the whole 10 s clip -> 'unlimited lookahead' upper bound). 50 Hz, float16."""
    import soundfile as sf
    from transformers import HubertModel
    todo = [c for c in ids if not os.path.exists(f'{FR}/{c}/au_ctx.npy') and os.path.exists(f'{DATA}/clips/{c}/audio_pad.wav')]
    if not todo: return
    m = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(dev).eval()
    for i in range(0, len(todo), 16):
        ch = todo[i:i + 16]
        w = []
        for c in ch:
            x, sr = sf.read(f'{DATA}/clips/{c}/audio_pad.wav', dtype='float32'); assert sr == 16000
            if x.ndim > 1: x = x.mean(1)
            x = np.pad(x, (0, max(0, 160000 - len(x))))[:160000]
            w.append((x - x.mean()) / (x.std() + 1e-7))
        w = torch.from_numpy(np.stack(w)).to(dev)
        o = m(w, output_hidden_states=True)
        loc = m.feature_extractor(w).transpose(1, 2)            # B,499,512 local conv features
        loc = torch.nn.functional.layer_norm(loc, (512,))
        ctx = o.hidden_states[9]
        for k, c in enumerate(ch):
            os.makedirs(f'{FR}/{c}', exist_ok=True)
            np.save(f'{FR}/{c}/au_loc.npy', loc[k].cpu().numpy().astype(np.float16))
            np.save(f'{FR}/{c}/au_ctx.npy', ctx[k].cpu().numpy().astype(np.float16))
    del m


WIN_LA = [0, 2, 5, 10]     # lookahead settings (video frames) for the causal windowed HuBERT features
WIN_TOK = 24


@torch.no_grad()
def prep_audio_win(ids, dev='cuda'):
    """Honestly causal contextual features: for video frame t and lookahead L, HuBERT-base runs on the 2.0 s of audio that ENDS
    at (end of frame t + L/30 s) and nothing later; the last 16 layer-9 tokens (0.32 s) are kept. -> au_win.npy (240,4,16,768) f16."""
    import soundfile as sf
    from transformers import HubertModel
    todo = [c for c in ids if not os.path.exists(f'{FR}/{c}/au_win.npy') and os.path.exists(f'{DATA}/clips/{c}/audio_pad.wav')]
    if not todo: return
    m = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(dev).eval().half()
    for c in todo:
        x, sr = sf.read(f'{DATA}/clips/{c}/audio_pad.wav', dtype='float32')
        if x.ndim > 1: x = x.mean(1)
        x = np.pad(x, (32000, max(0, 160000 - len(x)) + 16000))      # 2 s of zeros in front so every window is full length
        out = np.zeros((N, len(WIN_LA), WIN_TOK, 768), np.float16)
        for li, L in enumerate(WIN_LA):
            ends = [int(round(((t + 1 + L) / 30 + 1.0) * 16000)) + 32000 for t in range(N)]
            w = np.stack([x[e - 32000:e] for e in ends])
            w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-5)
            h = m(torch.from_numpy(w).to(dev).half(), output_hidden_states=True).hidden_states[9]
            out[:, li] = h[:, -WIN_TOK:].float().cpu().numpy()
        os.makedirs(f'{FR}/{c}', exist_ok=True)
        np.save(f'{FR}/{c}/au_win.npy', out)
    del m


def ready_ids(splits, audio='ctx'):
    out = []
    for c in manifest():
        if c['split'] in splits and all(os.path.exists(p) for p in (f"{FR}/{c['id']}/ok", f"{FR}/{c['id']}/au_{audio}.npy", f"{DATA}/clips/{c['id']}/lm.npz")):
            out.append(c)
    return out


class Clip:
    def __init__(self, cid, audio='loc'):
        z = np.load(f'{DATA}/clips/{cid}/lm.npz')
        self.id = cid
        face = z['face'].astype(np.float32); pose = z['pose'].astype(np.float32)
        fok = z['face_ok'].astype(bool); pok = z['pose_ok'].astype(bool)
        self.fok = fok.copy()
        # hold last good value through gaps
        for arr, ok in ((face, fok), (pose, pok)):
            last = None
            first = np.argmax(ok) if ok.any() else 0
            for t in range(len(arr)):
                if ok[t]: last = arr[t]
                elif last is not None: arr[t] = last
                else: arr[t] = arr[first]
        pose[~pok.any() if False else slice(0, 0)] = 0
        if not pok.any(): pose[:] = 0
        self.face, self.pose = face, pose
        self.au = np.load(f'{FR}/{cid}/au_{audio}.npy', mmap_mode='r' if audio == 'win' else None)
        self.win = audio == 'win'
        self.T = min(N, len(face))

    def window(self, t):
        idx = np.clip(np.arange(t - LM_PAST, t + LM_FUT + 1), 0, self.T - 1)
        if self.win: return self.face[idx].copy(), self.pose[idx].copy(), np.asarray(self.au[t]).astype(np.float32)
        k = int(round(((t + 0.5) / 30 + 1.0) * 50 - 0.5))
        ai = np.arange(k - AU_PAST, k + AU_FUT + 1)
        au = self.au[np.clip(ai, 0, len(self.au) - 1)].astype(np.float32)
        au[(ai < 0) | (ai >= len(self.au))] = 0
        return self.face[idx].copy(), self.pose[idx].copy(), au


def rd(p):
    return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)


def to_t(im):
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(127.5).sub(1)


def rand_aff(S=384, strong=False):
    s = random.uniform(0.94, 1.08); ang = random.uniform(-3, 3)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, s)
    M[0, 2] += random.uniform(-0.04, 0.04) * S; M[1, 2] += random.uniform(-0.04, 0.04) * S
    return M


def aff_xy(xy01, M, S=384):
    p = xy01 * S
    q = p @ M[:, :2].T + M[:, 2]
    return q / S


class DS(torch.utils.data.Dataset):
    def __init__(self, clips, audio='loc', length=10 ** 7, aug=True, noimg=False, seq=False):
        self.meta = clips; self.audio = audio; self.length = length; self.aug = aug; self.noimg = noimg; self.seq = seq
        self.mf = {}
        self.by_id = {}
        for c in clips: self.by_id.setdefault(c['identity'], []).append(c['id'])
        self.cache = {}

    def clip(self, cid):
        if cid not in self.cache: self.cache[cid] = Clip(cid, self.audio)
        return self.cache[cid]

    def __len__(self): return self.length

    def getseq(self):
        """5 frames at 25 fps (SyncNet window) with one shared augmentation + reference, plus MFCC positives / negatives."""
        while True:
            m = random.choice(self.meta); c = self.clip(m['id'])
            if c.id not in self.mf:
                p = f'{FR}/{c.id}/mfcc.npy'; self.mf[c.id] = np.load(p) if os.path.exists(p) else None
            mf = self.mf[c.id]
            if mf is None: continue
            nv = min(int(c.T / 1.2), mf.shape[1] // 4) - 6
            v = random.randrange(0, nv)
            ts = [min(int(round((v + k) * 1.2)), c.T - 1) for k in range(5)]
            if all(c.fok[t] for t in ts): break
        others = [x for x in self.by_id[m['identity']] if x != c.id]
        if others and random.random() < 0.7: rc = random.choice(others); rt = random.randrange(0, N)
        else: rc = c.id; rt = (ts[0] + random.randrange(60, N - 60)) % N
        rp = f'{FR}/{rc}/{rt:04d}.jpg'
        if not os.path.exists(rp): rc, rt, rp = c.id, ts[0], f'{FR}/{c.id}/{ts[0]:04d}.jpg'
        ref = rd(rp); rcl = self.clip(rc); rt = min(rt, rcl.T - 1)
        M = rand_aff()
        ref = cv2.warpAffine(ref, rand_aff(), (384, 384), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        out = {k: [] for k in ('y', 'face_w', 'pose_w', 'audio')}
        for t in ts:
            y = cv2.warpAffine(rd(f'{FR}/{c.id}/{t:04d}.jpg'), M, (384, 384), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fw, pw, au = c.window(t)
            fw[..., :2] = aff_xy(fw[..., :2], M); pw[..., :2] = aff_xy(pw[..., :2], M)
            out['y'].append(to_t(y)); out['face_w'].append(torch.from_numpy(fw)); out['pose_w'].append(torch.from_numpy(pw)); out['audio'].append(torch.from_numpy(au))
        out = {k: torch.stack(v_) for k, v_ in out.items()}
        r = to_t(ref)
        out['ref'] = r[None].expand(5, -1, -1, -1).clone()
        out['ref_face'] = torch.from_numpy(rcl.face[rt].copy())[None].expand(5, -1, -1).clone()
        out['ref_pose'] = torch.from_numpy(rcl.pose[rt].copy())[None].expand(5, -1, -1).clone()
        out['mf_pos'] = torch.from_numpy(mf[:, v * 4:v * 4 + 20].copy())
        negs = []
        while len(negs) < 6:
            u = random.randrange(0, nv)
            if abs(u - v) >= 8: negs.append(mf[:, u * 4:u * 4 + 20].copy())
        out['mf_neg'] = torch.from_numpy(np.stack(negs))
        return out

    def __getitem__(self, i):
        if self.seq: return self.getseq()
        while True:
            m = random.choice(self.meta); c = self.clip(m['id'])
            t = random.randrange(0, c.T)
            if c.fok[t]: break
        if self.noimg:
            others = [x for x in self.by_id[m['identity']] if x != c.id]
            rc = random.choice(others) if others and random.random() < 0.7 else c.id
            rcl = self.clip(rc); rt = random.randrange(0, rcl.T)
            fw, pw, au = c.window(t)
            if self.aug:
                M = rand_aff(); fw[..., :2] = aff_xy(fw[..., :2], M); pw[..., :2] = aff_xy(pw[..., :2], M)
            return {'ref_face': torch.from_numpy(rcl.face[rt].copy()), 'ref_pose': torch.from_numpy(rcl.pose[rt].copy()),
                    'face_w': torch.from_numpy(fw), 'pose_w': torch.from_numpy(pw), 'audio': torch.from_numpy(au)}
        y = rd(f'{FR}/{c.id}/{t:04d}.jpg')
        # identity reference: another clip of the same person when there is one, else a frame >= 2 s away
        others = [x for x in self.by_id[m['identity']] if x != c.id]
        if others and random.random() < 0.7:
            rc = random.choice(others); rt = random.randrange(0, N)
        else:
            rc = c.id; rt = (t + random.randrange(60, N - 60)) % N
        rp = f'{FR}/{rc}/{rt:04d}.jpg'
        ref = rd(rp) if os.path.exists(rp) else y
        fw, pw, au = c.window(t)
        if self.aug:
            M = rand_aff()
            y = cv2.warpAffine(y, M, (384, 384), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fw[..., :2] = aff_xy(fw[..., :2], M); pw[..., :2] = aff_xy(pw[..., :2], M)
            ref = cv2.warpAffine(ref, rand_aff(), (384, 384), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if random.random() < 0.5:
                ref = np.clip((ref.astype(np.float32) - 128) * random.uniform(0.92, 1.08) + 128 + random.uniform(-8, 8), 0, 255).astype(np.uint8)
        rcl = self.clip(rc); rt = min(rt, rcl.T - 1)
        return {'y': to_t(y), 'ref': to_t(ref), 'ref_face': torch.from_numpy(rcl.face[rt].copy()), 'ref_pose': torch.from_numpy(rcl.pose[rt].copy()), 'face_w': torch.from_numpy(fw), 'pose_w': torch.from_numpy(pw), 'audio': torch.from_numpy(au)}


def lower_face_idx(clips, n=40):
    """Points of the lower face (below the nose base, pt 2) in the mean face-normalised shape: the set hidden in 'audio-lips' mode."""
    acc = []
    for c in clips[:n]:
        z = np.load(f"{DATA}/clips/{c['id']}/lm.npz"); ok = z['face_ok'].astype(bool)
        if ok.sum() < 10: continue
        f = z['face'][ok][::10, :, :2].astype(np.float64)
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
