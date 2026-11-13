"""torso_up pose2real-TORSO (port of pose2real): landmarks + skeleton + HANDS + audio (+ identity reference) -> photoreal frame,
one diffusion step, 512x640 (W x H) torso-up frames (any W,H multiple of 64).

Owner's design: a deep trainable encoder for landmarks/skeleton, an audio encoder (pretrained speech features +
trainable layers), mixed together, and the mixed embedding is ADDED TO THE INITIAL NOISE of a pretrained diffusion
model (SD-Turbo, one step). Ablation switches: injection (noise | early | controlnet | xattn), encoder
(splat | vec | raster), depth, audio / landmark lookahead, audio features, backbone, hands (tokens | none).
Coordinates: x/width, y/height in [0,1] (per axis). Anything measuring distances goes through pixels.
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import AutoTokenizer, CLIPTextModel
from peft import LoraConfig

PROMPT = 'a high quality photo of a real person talking to the camera, natural skin, photorealistic'
BACKBONES = {
    'sd-turbo': dict(repo='stabilityai/sd-turbo', variant='fp16'),
    'sd15': dict(repo='stable-diffusion-v1-5/stable-diffusion-v1-5', variant='fp16'),
    'tiny-sd': dict(repo='segmind/tiny-sd', variant=None),
}
UNET_T = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_shortcut", "conv_out",
          "proj_in", "proj_out", "ff.net.2", "ff.net.0.proj"]

NF, NP, NH = 478, 33, 42       # face, pose, hand points (hands: 2 x 21, index 0 = anatomical LEFT, 1 = RIGHT)
# pose bones that can show in a head-and-shoulders crop (MediaPipe pose indices)
BONES = [(11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (11, 23), (12, 24), (23, 24), (9, 10), (7, 3), (8, 6), (0, 11), (0, 12)]
# the 21 standard MediaPipe hand connections (per hand), and pose wrist (15 left / 16 right) -> hand wrist (point 0)
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14),
              (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
HMC = 10                        # raster heat-map channels: 8 face/pose + hand points + hand bones (zeros with --hands none)


def token_bones(hands=True):
    """All splatted bones as (token index a, token index b) with tokens ordered face, pose, hands."""
    bl = [(NF + a, NF + b) for a, b in BONES]
    if hands:
        H0 = NF + NP
        for h in range(2):
            bl += [(H0 + 21 * h + a, H0 + 21 * h + b) for a, b in HAND_BONES]
        bl += [(NF + 15, H0), (NF + 16, H0 + 21)]
    return bl
LM_PAST, LM_FUT = 2, 5          # landmark window stored per sample: t-2 .. t+5
AU_PAST, AU_FUT = 25, 17        # audio steps (50 Hz): 0.5 s back, 0.34 s ahead
# MediaPipe lips (FACEMESH_LIPS vertices) for the mouth loss / heatmap group
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14,
        317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415]
LEYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
REYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
IRIS = list(range(468, 478))


def splat(xy, feat, w, rh, rw):
    """Bilinear scatter. xy (B,N,2) in [0,1] (x of width, y of height), feat (B,N,C), w (B,N) -> (B,C,rh,rw) float32."""
    feat = feat.float(); xy = xy.float(); w = w.float()
    B, N, C = feat.shape
    p = xy * xy.new_tensor([rw, rh]) - 0.5
    x0 = p[..., 0].floor(); y0 = p[..., 1].floor()
    fx = p[..., 0] - x0; fy = p[..., 1] - y0
    out = feat.new_zeros(B, rh * rw, C)
    for dx, dy, ww in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = x0 + dx; yi = y0 + dy
        ok = ((xi >= 0) & (xi < rw) & (yi >= 0) & (yi < rh)).to(feat.dtype)
        idx = (yi.clamp(0, rh - 1) * rw + xi.clamp(0, rw - 1)).long()
        out.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, C), feat * (ww * ok * w).unsqueeze(-1))
    return out.transpose(1, 2).reshape(B, C, rh, rw)


def _lines(a, b, va, vb, n):
    """Points along segments a->b (B,K,2), visibility min(va,vb) (B,K) -> (B,K*n,2), (B,K*n)."""
    ts = torch.linspace(0, 1, n, device=a.device).view(1, 1, -1, 1)
    pts = (a[:, :, None].float() * (1 - ts) + b[:, :, None].float() * ts).reshape(a.shape[0], -1, 2)
    return pts, torch.minimum(va, vb).float()[:, :, None].expand(-1, -1, n).reshape(a.shape[0], -1)


def heatmap(face, pose, hands, rh, rw):
    """Fixed (non-learned) drawing of the conditioning: HMC=10 channels at (rh, rw). face (B,478,>=2) xy in [0,1];
    pose (B,33,4) (vis in [...,3]); hands (B,2,21,4) with ok in [...,3], or None (-> the two hand channels stay zero)."""
    B = face.shape[0]
    ch = []
    groups = [LIPS, LEYE + REYE, BROWS, OVAL, IRIS]
    one = face.new_ones(B, 1)
    for g in groups:
        ch.append(splat(face[:, g, :2], one.expand(B, len(g)).unsqueeze(-1), one.expand(B, len(g)), rh, rw))
    ch.append(splat(face[:, :468, :2], face.new_ones(B, 468, 1), face.new_ones(B, 468), rh, rw) * 0.5)
    pv = (pose[..., 3] > 0.5).float()
    ia = [b[0] for b in BONES]; ib = [b[1] for b in BONES]
    pts, vis = _lines(pose[:, ia, :2], pose[:, ib, :2], pv[:, ia], pv[:, ib], 12)
    ch.append(splat(pts, pts.new_ones(B, pts.shape[1], 1), vis, rh, rw))
    ch.append(splat(pose[:, :, :2], pose.new_ones(B, NP, 1), pv, rh, rw))
    if hands is None:
        ch.append(face.new_zeros(B, 2, rh, rw).float())
    else:
        hx = torch.cat([pose[:, :, :2], hands[..., :2].reshape(B, NH, 2)], 1)      # pose + hand points, one index space
        hv = torch.cat([pv, hands[..., 3].reshape(B, NH).float()], 1)
        tb = [(a - NF, b - NF) for a, b in token_bones(True)[len(BONES):]]          # hand bones + wrist links
        ja = [x[0] for x in tb]; jb = [x[1] for x in tb]
        ch.append(splat(hx[:, NP:], hx.new_ones(B, NH, 1), hv[:, NP:], rh, rw))
        pts, vis = _lines(hx[:, ja], hx[:, jb], hv[:, ja], hv[:, jb], 6)
        ch.append(splat(pts, pts.new_ones(B, pts.shape[1], 1), vis, rh, rw))
    return torch.cat(ch, 1).clamp(0, 2)


class Res(nn.Module):
    def __init__(self, c, gdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, c); self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.n2 = nn.GroupNorm(8, c); self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.g = nn.Linear(gdim, 2 * c)

    def forward(self, x, g):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.g(g).chunk(2, -1)
        h = self.n2(h) * (1 + s[..., None, None]) + b[..., None, None]
        return x + self.c2(F.silu(h))


class CondEncoder(nn.Module):
    """Landmarks + skeleton + hands + audio -> mixed embedding (spatial map(s) and/or tokens).
    size = (W, H) of the image; the splat map is (H/4, W/4), the output map (H/8, W/8) = latent size."""

    def __init__(self, enc='splat', depth=6, d=384, heads=6, audio_dim=512, cs=96, inj='noise', ctx_dim=1024,
                 unet_ch=(320, 640, 1280, 1280), lower_idx=None, refbase=False, hands='tokens', size=(512, 640)):
        super().__init__()
        assert hands in ('tokens', 'none') and size[0] % 64 == 0 and size[1] % 64 == 0, (hands, size)
        self.enc, self.inj, self.d, self.cs = enc, inj, d, cs
        self.refbase = refbase
        self.rw, self.rh = size[0] // 4, size[1] // 4
        self.nh = NH if hands == 'tokens' else 0
        N = self.N = NF + NP + self.nh
        W = LM_PAST + 1 + LM_FUT
        self.W = W
        self.pid = nn.Parameter(torch.randn(N, d) * 0.02)
        self.pt_in = nn.Sequential(nn.Linear(W * 4 + 3, d), nn.SiLU(), nn.Linear(d, d))
        self.mask_emb = nn.Parameter(torch.zeros(d))
        self.au_in = nn.Sequential(nn.Linear(audio_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.au_pos = nn.Parameter(torch.randn(AU_PAST + 1 + AU_FUT, d) * 0.02)
        self.glob = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.la_emb = nn.Embedding(11, d); nn.init.zeros_(self.la_emb.weight)
        self.qh, self.qw = size[1] // 32, size[0] // 32
        self.nq = self.qh * self.qw if enc == 'vec' else 0
        if self.nq: self.q = nn.Parameter(torch.randn(1, self.nq, d) * 0.02)
        if enc != 'raster':
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        else:  # no deep point encoder: only audio gets a small transformer, points are drawn as fixed heat maps
            layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            self.hm_in = nn.Conv2d(HMC, cs, 3, padding=1)
        self.norm = nn.LayerNorm(d)
        self.xy_head = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2))   # predicts masked points (audio-driven lips)
        self.pt_out = nn.Linear(d, cs)
        self.bones = token_bones(self.nh > 0)
        self.bone_emb = nn.Parameter(torch.randn(len(self.bones), cs) * 0.02)
        if enc == 'vec':
            self.q_out = nn.Linear(d, cs)
            self.up = nn.Sequential(nn.ConvTranspose2d(cs, cs, 4, 2, 1), nn.SiLU(), nn.ConvTranspose2d(cs, cs, 4, 2, 1), nn.SiLU(),
                                    nn.ConvTranspose2d(cs, cs, 4, 2, 1))
        self.stem = nn.Conv2d(cs + 1, cs, 3, padding=1)
        self.r96 = nn.ModuleList([Res(cs, d) for _ in range(2)])
        self.down = nn.Conv2d(cs, 2 * cs, 3, stride=2, padding=1)
        self.r48 = nn.ModuleList([Res(2 * cs, d) for _ in range(3)])
        c = 2 * cs
        if inj == 'noise':
            self.head = nn.Conv2d(c, 4, 3, padding=1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        elif inj == 'early':
            self.head = nn.Conv2d(c, unet_ch[0], 3, padding=1); nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)
        elif inj == 'controlnet':
            self.pyr = nn.ModuleList(); self.pyr_res = nn.ModuleList(); cin = c
            for i, uc in enumerate(unet_ch):
                self.pyr.append(nn.Conv2d(cin, uc // 2, 3, stride=1 if i == 0 else 2, padding=1)); cin = uc // 2
                self.pyr_res.append(Res(cin, d))
            # SD down residual layout: [c0@r, c0@r, c0@r, c0@r/2, c1, c1, c1@r/4, c2, c2, c2@r/8, c3, c3] + mid
            self.lay = [(0, unet_ch[0])] * 3 + [(1, unet_ch[0]), (1, unet_ch[1]), (1, unet_ch[1]), (2, unet_ch[1]), (2, unet_ch[2]),
                                                (2, unet_ch[2]), (3, unet_ch[2]), (3, unet_ch[3]), (3, unet_ch[3]), (3, unet_ch[3])]
            self.zero = nn.ModuleList([nn.Conv2d(unet_ch[l] // 2, oc, 1) for l, oc in self.lay])
            for z in self.zero: nn.init.zeros_(z.weight); nn.init.zeros_(z.bias)
        if inj == 'xattn':
            self.tok_out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, ctx_dim))
        low = torch.zeros(N, dtype=torch.bool)
        if lower_idx is not None: low[:len(lower_idx)] = lower_idx[:N]
        self.register_buffer('lower', low)

    def forward(self, b, lm_la=0, au_la=5, mask_lower=None, xy_only=False):
        """b: face_w (B,W,478,3) xy01+z, pose_w (B,W,33,4), hands_w (B,W,2,21,4) xy01+z+ok, audio (B,A,adim).
        mask_lower (B,) bool: hide lower-face points. Returns dict(map=..., down/mid, tokens, xy_pred, xy_true, hide)."""
        face_w, pose_w, au = b['face_w'], b['pose_w'], b['audio']
        B = face_w.shape[0]; dev = face_w.device; N = self.N; rh, rw = self.rh, self.rw
        if mask_lower is None: mask_lower = torch.zeros(B, dtype=torch.bool, device=dev)
        W = self.W
        # per-point trajectories over the window (absolute, centred); future beyond lookahead masked
        tmask = torch.zeros(W, device=dev); tmask[:LM_PAST + 1 + lm_la] = 1
        f = torch.cat([(face_w[..., :2] - 0.5) * 2, face_w[..., 2:3] * 2, torch.ones_like(face_w[..., :1])], -1)     # B,W,478,4
        p = torch.cat([(pose_w[..., :2] - 0.5) * 2, pose_w[..., 2:3].clamp(-2, 2) * 0, pose_w[..., 3:4]], -1)       # B,W,33,4 (z dropped, vis kept)
        parts = [f, p]
        cur = [face_w[:, LM_PAST, :, :2], pose_w[:, LM_PAST, :, :2]]                                              # true xy01
        vis = [torch.ones(B, NF, device=dev), (pose_w[:, LM_PAST, :, 3] > 0.5).float()]
        hands = None
        if self.nh:
            hw = b['hands_w'].reshape(B, W, NH, 4)
            ok = hw[..., 3:4]
            parts.append(torch.cat([(hw[..., :2] - 0.5) * 2, (hw[..., 2:3] * 2).clamp(-2, 2), torch.ones_like(ok)], -1) * ok)  # missing hand = 0
            cur.append(hw[:, LM_PAST, :, :2]); vis.append(hw[:, LM_PAST, :, 3].float())
            hands = b['hands_w'][:, LM_PAST]
        traj = torch.cat(parts, 2) * tmask.view(1, W, 1, 1)                                                     # B,W,N,4
        traj = traj.permute(0, 2, 1, 3).reshape(B, N, W * 4)
        cur = torch.cat(cur, 1); vis = torch.cat(vis, 1)
        hide = self.lower.view(1, -1) & mask_lower.view(B, 1)                                                   # B,N
        extra = torch.stack([hide.float(), vis, torch.full_like(vis, lm_la / LM_FUT)], -1)
        traj = traj * (~hide).unsqueeze(-1)
        base = None
        if self.refbase and 'ref_face' in b:
            # hidden lower-face points start from the REFERENCE frame's own lower face, carried into the current head pose by an
            # affine fit on the visible upper-face points; the network then only has to predict the articulation offset from audio.
            rxy = b['ref_face'][..., :2].float()                                                                # B,NF,2
            up = ~self.lower[:NF]
            X = torch.cat([rxy[:, up], torch.ones(B, int(up.sum()), 1, device=dev)], -1)
            A = torch.linalg.lstsq(X, cur[:, :NF][:, up].float()).solution                                      # B,3,2
            base = torch.cat([torch.cat([rxy, torch.ones(B, NF, 1, device=dev)], -1) @ A, cur[:, NF:].float()], 1)  # only face is ever hidden
            scale = (cur[:, 263] - cur[:, 33]).float().norm(dim=-1).view(B, 1, 1)                               # inter-ocular
            bfeat = torch.zeros_like(traj); bfeat[..., LM_PAST * 4:LM_PAST * 4 + 2] = (base - 0.5) * 2
            traj = torch.where(hide.unsqueeze(-1), bfeat.to(traj.dtype), traj)
        x = self.pt_in(torch.cat([traj, extra], -1)) + self.pid + hide.unsqueeze(-1) * self.mask_emb
        A = au.shape[1]
        amask = torch.zeros(A, dtype=torch.bool, device=dev)
        amask[AU_PAST + 1 + int(round(au_la / 30 * 50)):] = True                                                # True = not visible
        a = self.au_in(au) + self.au_pos[:A] + self.la_emb.weight[min(int(au_la), 10)]
        toks = [self.glob.expand(B, -1, -1), a]
        kpm = [torch.zeros(B, 1, dtype=torch.bool, device=dev), amask.view(1, A).expand(B, A)]
        if self.enc != 'raster':
            toks.append(x); kpm.append(torch.zeros(B, N, dtype=torch.bool, device=dev))
            if self.nq: toks.append(self.q.expand(B, -1, -1)); kpm.append(torch.zeros(B, self.nq, dtype=torch.bool, device=dev))
        h = self.norm(self.tf(torch.cat(toks, 1), src_key_padding_mask=torch.cat(kpm, 1)))
        g = h[:, 0]
        out = {'xy_true': cur, 'hide': hide}
        if self.enc == 'raster':
            fz = face_w[:, LM_PAST]
            if mask_lower.any():   # hidden lower face: just not drawn (audio can only modulate through FiLM)
                fz = torch.where(hide[:, :NF].unsqueeze(-1), torch.full_like(fz, -1.0), fz)
            hm = heatmap(fz, pose_w[:, LM_PAST], hands, rh, rw)
            m = self.hm_in(hm.to(self.hm_in.weight.dtype))
            cnt = hm.sum(1, keepdim=True).clamp(0, 1)
            out['xy_pred'] = cur
        elif self.enc == 'vec':
            hp = h[:, 1 + A:1 + A + N]
            out['xy_pred'] = torch.sigmoid(self.xy_head(hp) * 0.1 + torch.logit(cur.clamp(0.02, 0.98)) * (~hide).unsqueeze(-1))
            q = self.q_out(h[:, -self.nq:]).transpose(1, 2).reshape(B, self.cs, self.qh, self.qw)
            m = self.up(q); cnt = torch.ones(B, 1, rh, rw, device=dev)
        else:
            hp = h[:, 1 + A:1 + A + N]
            if base is not None:
                xy_pred = base + torch.tanh(self.xy_head(hp).float()) * 0.5 * scale
            else:
                xy_pred = torch.sigmoid(self.xy_head(hp))
            out['xy_pred'] = xy_pred
            if xy_only: return out
            xy = torch.where(hide.unsqueeze(-1), xy_pred, cur)
            pf = self.pt_out(hp)
            m = splat(xy, pf, vis, rh, rw)
            ia = [bb[0] for bb in self.bones]; ib = [bb[1] for bb in self.bones]
            ts = torch.linspace(0, 1, 10, device=dev).view(1, 1, -1, 1)
            bp = (xy[:, ia, None] * (1 - ts) + xy[:, ib, None] * ts).reshape(B, -1, 2)
            bf = ((pf[:, ia] + pf[:, ib]) * 0.5 + self.bone_emb)[:, :, None].expand(-1, -1, 10, -1).reshape(B, -1, self.cs)
            bv = torch.minimum(vis[:, ia], vis[:, ib])[:, :, None].expand(-1, -1, 10).reshape(B, -1)
            m = m + splat(bp, bf, bv, rh, rw)
            cnt = splat(xy, torch.ones(B, N, 1, device=dev), vis, rh, rw).clamp(0, 2)
        m = self.stem(torch.cat([m, cnt.to(m.dtype)], 1))
        for r in self.r96: m = r(m, g)
        m = self.down(m)
        for r in self.r48: m = r(m, g)
        if self.inj in ('noise', 'early'):
            out['map'] = self.head(m)
        elif self.inj == 'controlnet':
            feats = []
            for cv, rr in zip(self.pyr, self.pyr_res):
                m = rr(F.silu(cv(m)), g); feats.append(m)
            rs = [z(feats[l]) for z, (l, _) in zip(self.zero, self.lay)]
            out['down'] = rs[:12]; out['mid'] = rs[12]
        elif self.inj == 'xattn':
            out['tokens'] = self.tok_out(torch.cat([h[:, :1], h[:, 1:1 + A][:, ~amask], h[:, 1 + A:1 + A + N]], 1)
                                         if self.enc != 'raster' else h[:, :1 + A][:, ~torch.cat([amask[:1] & False, amask])])
        return out


class P2R(nn.Module):
    def __init__(self, backbone='sd-turbo', inj='noise', enc='splat', depth=6, audio_dim=512, rank=32, lower_idx=None, ref=True,
                 full_ft=False, refbase=False, hands='tokens', size=(512, 640)):
        super().__init__()
        bb = BACKBONES[backbone]; repo = bb['repo']; kw = dict(variant=bb['variant']) if bb['variant'] else {}
        tok = AutoTokenizer.from_pretrained(repo, subfolder='tokenizer')
        te = CLIPTextModel.from_pretrained(repo, subfolder='text_encoder', torch_dtype=torch.float32, **kw)
        with torch.no_grad():
            ids = tok(PROMPT, max_length=tok.model_max_length, padding='max_length', truncation=True, return_tensors='pt').input_ids
            self.register_buffer('emb', te(ids)[0].float()[:, :24].clone(), persistent=False)
        del te
        sched = DDPMScheduler.from_pretrained(repo, subfolder='scheduler')
        self.pred_type = sched.config.prediction_type
        self.ab = float(sched.alphas_cumprod[999])
        self.vae = AutoencoderKL.from_pretrained(repo, subfolder='vae', torch_dtype=torch.float32, **kw).requires_grad_(False)
        unet = UNet2DConditionModel.from_pretrained(repo, subfolder='unet', torch_dtype=torch.float32, **kw)
        unet.requires_grad_(full_ft)
        if not full_ft:
            unet.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights='gaussian', target_modules=UNET_T))
        self.use_ref = ref
        old = unet.conv_in; base = old.base_layer if hasattr(old, 'base_layer') else old
        new = nn.Conv2d(8 if ref else 4, base.out_channels, 3, padding=1)
        with torch.no_grad():
            new.weight.zero_(); new.weight[:, :4] = base.weight; new.bias.copy_(base.bias)
        unet.conv_in = new; unet.conv_in.requires_grad_(True)
        self.unet = unet; self.sf = self.vae.config.scaling_factor
        ctx = unet.config.cross_attention_dim; self.ctx = ctx
        ch = tuple(unet.config.block_out_channels)
        self.inj = inj
        self.cond = CondEncoder(enc=enc, depth=depth, audio_dim=audio_dim, inj=inj, ctx_dim=ctx,
                                unet_ch=ch if len(ch) == 4 else (ch + ch[-1:] * 4)[:4], lower_idx=lower_idx, refbase=refbase,
                                hands=hands, size=size)
        self.size = tuple(size)   # (W, H); latent (H/8, W/8)
        if ref:  # identity reference tokens for cross-attention ((H/32)x(W/32) patches of the reference latent)
            self.ref_tok = nn.Sequential(nn.Conv2d(4, 256, 3, padding=1), nn.SiLU(), nn.Conv2d(256, 512, 4, stride=4), nn.SiLU(),
                                         nn.Conv2d(512, ctx, 1))
            self.ref_pos = nn.Parameter(torch.randn(1, (size[1] // 32) * (size[0] // 32), ctx) * 0.02)
        self._early = None
        if inj == 'early':
            self.unet.conv_in.register_forward_hook(lambda m, i, o: o + self._early.to(o.dtype) if self._early is not None else o)
        self.full_ft = full_ft

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def set_ckpt(self, on=True):
        self.vae.enable_gradient_checkpointing()
        if on: self.unet.enable_gradient_checkpointing()

    @torch.no_grad()
    def encode(self, x):
        return self.vae.encode(x).latent_dist.mode() * self.sf

    def forward(self, b, noise, ref_lat=None, lm_la=0, au_la=5, mask_lower=None, decode=True):
        c = self.cond(b, lm_la=lm_la, au_la=au_la, mask_lower=mask_lower)
        B = noise.shape[0]
        z = noise
        kw = {}
        if self.inj == 'noise':
            z = noise + c['map'].to(noise.dtype)          # <- the owner's proposal: embedding added to the initial noise
        elif self.inj == 'early':
            self._early = c['map']
        elif self.inj == 'controlnet':
            kw = dict(down_block_additional_residuals=tuple(r.to(noise.dtype) for r in c['down']), mid_block_additional_residual=c['mid'].to(noise.dtype))
        ctx = [self.emb.expand(B, -1, -1).to(noise.dtype)]
        zin = z
        if self.use_ref:
            ctx.append((self.ref_tok(ref_lat).flatten(2).transpose(1, 2) + self.ref_pos).to(noise.dtype))
            zin = torch.cat([z, ref_lat], 1)
        if self.inj == 'xattn':
            ctx.append(c['tokens'].to(noise.dtype))
        t = torch.full((B,), 999, device=noise.device, dtype=torch.long)
        pred = self.unet(zin, t, encoder_hidden_states=torch.cat(ctx, 1), **kw).sample
        self._early = None
        a = self.ab
        if self.pred_type == 'v_prediction':
            x0 = a ** 0.5 * z - (1 - a) ** 0.5 * pred
        else:
            x0 = (z - (1 - a) ** 0.5 * pred) / a ** 0.5
        c['x0'] = x0
        if decode:
            c['img'] = self.vae.decode(x0 / self.sf).sample
        return c

    def trainable_state(self):
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep or k.startswith('cond.lower')}

    def load_trainable(self, sd):
        r = self.load_state_dict(sd, strict=False)
        assert not r.unexpected_keys, r.unexpected_keys[:5]
