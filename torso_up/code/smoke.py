"""CPU smoke test (real SD-Turbo weights from ./hf, bf16 autocast on CPU).
    venv/bin/python smoke.py            # synthetic torso clips built from PRIOR bust clips (scratch/), every combination
    venv/bin/python smoke.py --real     # the DATA agent's real heldout clips (ROOT/data, after HELDOUT_DONE)
Runs prep.py (frames + HuBERT loc/win on CPU), unit checks of the data pipeline (clips with no hands at all, the hand-frame
sampler, both sizes), 2 training steps per combination with every loss on (l2, LPIPS, mouth, hand, GAN, hand GAN, xy), and
infer on 8 frames asserting a 512x640 x 8-frame video."""
import argparse, os, sys, json, subprocess, time, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PRIOR = __import__('os').path.expandvars('$WORK/data')
ap = argparse.ArgumentParser()
ap.add_argument('--real', action='store_true')
ap.add_argument('--only', default='')        # comma list of combo names
ap.add_argument('--skip_prep', action='store_true')
ap.add_argument('--amp', default='none')   # matrix precision on CPU (this Broadwell CPU has no bf16 units: bf16 is emulated, ~10x slower); the bf16 combo always runs bf16
a = ap.parse_args()
SC = f'{HERE}/scratch' + ('_real' if a.real else '')
DATA = f'{ROOT}/data' if a.real else f'{SC}/data'
env = dict(os.environ, DATA=DATA, FRAMES=f'{SC}/frames', HF_HOME=f'{HERE}/hf', TORCH_HOME=f'{HERE}/hf/torch', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6',
           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
os.environ.update(env)
PY = sys.executable
S = 0.6 * 384; OX, OY = (512 - S) / 2, 40          # PRIOR 384 bust frame pasted into a grey 512x640 canvas


def run(cmd, log):
    t0 = time.time()
    with open(log, 'w') as f:
        r = subprocess.run(['nice', '-n', '10'] + cmd, cwd=HERE, env=env, stdout=f, stderr=subprocess.STDOUT)
    tail = open(log).read().strip().splitlines()[-3:]
    print(f"  {'OK ' if r.returncode == 0 else 'FAIL'} {time.time() - t0:5.0f}s  {' '.join(cmd[1:4])} ...  | {tail[-1] if tail else ''}", flush=True)
    if r.returncode: print('\n'.join(open(log).read().splitlines()[-25:])); sys.exit(1)


def synth():
    """5 clips: train x3 (hands both / one with hands_ok all 0 / one whose lm.npz has NO hand keys), heldout + heldout_ref of P0115."""
    import cv2
    spec = [('M000_P0115', 'heldout', 'same12', 'both'), ('R000_P0115', 'heldout_ref', 'prior', 'both'),
            ('M002_P0116', 'train', 'prior', 'both'), ('M004_P0116', 'train', 'gesture', 'zero'), ('M012_P0299A', 'train', 'prior', 'nokey')]
    man = []
    rng = np.random.default_rng(0)
    for cid, split, st, hk in spec:
        d = f'{DATA}/clips/{cid}'; os.makedirs(d, exist_ok=True)
        man.append({'id': cid, 'identity': cid.split('_')[1], 'split': split, 'set': st, 'origin': 'torso_up-synth', 'hand_frac': 0.0 if hk != 'both' else 0.6})
        if os.path.exists(f'{d}/lm.npz'): continue
        shutil.copy(f'{PRIOR}/clips/{cid}/audio_pad.wav', d)
        z = dict(np.load(f'{PRIOR}/clips/{cid}/lm.npz'))
        for k in ('face', 'pose'):   # bust coords -> torso canvas coords (per axis)
            z[k] = z[k].copy(); z[k][..., 0] = (OX + S * z[k][..., 0]) / 512; z[k][..., 1] = (OY + S * z[k][..., 1]) / 640
        T = len(z['face'])
        if hk != 'nokey':
            hands = np.zeros((T, 2, 21, 3), np.float32); ok = np.zeros((T, 2), bool)
            if hk == 'both':
                ang = np.linspace(-1.2, 1.2, 21)[None]
                for h, cx in ((0, 0.70), (1, 0.30)):   # anatomical left hand appears on the image right
                    on = (np.arange(T) // 40 + h) % 3 != 0
                    t = np.arange(T)[:, None] / 30
                    wx = cx + 0.05 * np.sin(2 * t + h); wy = 0.78 + 0.04 * np.cos(3 * t)
                    r = 0.03 + 0.04 * (np.arange(21)[None] % 4) / 3
                    hands[:, h, :, 0] = wx + r * np.sin(ang) * 0.8; hands[:, h, :, 1] = wy - r * np.cos(ang); hands[:, h, 0, :2] = np.c_[wx, wy]
                    hands[:, h, :, 2] = rng.normal(0, 0.02, (T, 21))
                    ok[:, h] = on; hands[~on, h] = 0
                    z['pose'][on, 15 + h, :2] = np.c_[wx, wy + 0.03][on]; z['pose'][on, 15 + h, 3] = 0.9
            z['hands'] = hands; z['hands_ok'] = ok; z['hands_score'] = ok * 0.9
        np.savez(f'{d}/lm.npz', **z)
        cap = cv2.VideoCapture(f'{PRIOR}/clips/{cid}/real.mp4')
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '512x640', '-r', '30', '-i', '-',
                              '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', f'{d}/real.mp4'], stdin=subprocess.PIPE)
        while True:
            ok_, fr = cap.read()
            if not ok_: break
            cv = np.full((640, 512, 3), 120, np.uint8); s = int(round(S))
            cv[int(OY):int(OY) + s, int(OX):int(OX) + s] = cv2.resize(fr, (s, s), interpolation=cv2.INTER_AREA)
            p.stdin.write(cv.tobytes())
        p.stdin.close(); p.wait()
    json.dump({'clips': man, 'heldout_identities': ['P0115']}, open(f'{DATA}/manifest.json', 'w'))


def unit():
    import torch, data as D, p2r
    m = D.manifest()
    for c in m:
        if not os.path.exists(f"{D.FR}/{c['id']}/ok"): continue
        cl = D.Clip(c['id'], 'loc'); fw, pw, hw, au = cl.window(0)
        assert fw.shape == (8, 478, 3) and pw.shape == (8, 33, 4) and hw.shape == (8, 2, 21, 4), (fw.shape, pw.shape, hw.shape)
        assert np.all(hw[..., :3][hw[..., 3] == 0] == 0)
        print(f"  clip {c['id']}: T {cl.T} hand frames {len(cl.hand_t)} au {au.shape}")
    tr = [c for c in D.ready_ids(['train', 'heldout'], 'loc')]
    for size in ((512, 640), (384, 512)):
        ds = D.DS(tr, 'loc', size=size, p_hand=0.5)
        n_h = 0
        for i in range(40):
            b = ds[i]
            assert b['y'].shape == (3, size[1], size[0]) and b['ref'].shape == b['y'].shape, b['y'].shape
            n_h += b['hands_w'][2, :, 0, 3].max().item() > 0
        print(f'  DS {size}: 40 samples ok, {n_h} with a visible hand at t (p_hand 0.5)')
        hm = p2r.heatmap(b['face_w'][None, 2], b['pose_w'][None, 2], b['hands_w'][None, 2], size[1] // 4, size[0] // 4)
        assert hm.shape == (1, p2r.HMC, size[1] // 4, size[0] // 4), hm.shape
    # augmentation keeps image and landmarks aligned: a dot drawn at a hand point stays under the transformed point
    import cv2, random
    random.seed(3); W, H = 512, 640
    im = np.zeros((H, W, 3), np.uint8); xy = np.array([[0.7, 0.8]], np.float32)
    cv2.circle(im, (int(xy[0, 0] * W), int(xy[0, 1] * H)), 3, (255, 255, 255), -1)
    M = D.rand_aff(W, H); im2 = cv2.warpAffine(im, M, (W, H)); q = D.aff_xy(xy, M, W, H)[0] * [W, H]
    ys, xs = np.nonzero(im2[..., 0] > 128); assert abs(xs.mean() - q[0]) < 1.5 and abs(ys.mean() - q[1]) < 1.5, (xs.mean(), ys.mean(), q)
    print('  augmentation alignment ok')


if not a.real: synth()
else: assert os.path.exists(f'{ROOT}/data/HELDOUT_DONE'), 'no HELDOUT_DONE yet'
os.makedirs(f'{SC}/logs', exist_ok=True); os.makedirs(f'{SC}/runs', exist_ok=True)
sp = 'heldout,heldout_ref,train'
if not a.skip_prep:
    print('prep (frames + HuBERT loc, win on CPU)', flush=True)
    run([PY, 'prep.py', '--dev', 'cpu', '--audio', 'loc,win', '--splits', sp, '--jobs', '4'] + (['--max_clips', '4'] if a.real else []), f'{SC}/logs/prep.log')
print('unit checks', flush=True); unit()
TS = 'heldout' if a.real else 'train'      # real: only the heldout clips exist this early -> train on them for the smoke test
base = ['--dev', 'cpu', '--bs', '2', '--max_steps', '2', '--gan_start', '0', '--workers', '1', '--hand_gan', '--w_hand', '2', '--p_hide', '0.5',
        '--splits', TS, '--depth', '2', '--minutes', '120']
combos = {}
for hands in ('tokens', 'none'):
    for enc in ('splat', 'raster'):
        for inj in ('noise', 'controlnet'):
            combos[f'{hands}_{enc}_{inj}'] = ['--hands', hands, '--enc', enc, '--inj', inj]
combos['s384_tokens_splat_noise'] = ['--size', '384x512']
ONE = ['--max_steps', '1']    # extras: one step each (CPU time)
combos['early_vec'] = ['--inj', 'early', '--enc', 'vec'] + ONE
combos['xattn'] = ['--inj', 'xattn'] + ONE
# bf16 autocast path as on the GPU; small frame because this CPU emulates bf16 (~10x slower than fp32)
combos['bf16_tokens_splat_noise'] = ['--amp', 'bf16', '--size', '192x256'] + ONE
combos['best_win_refbase'] = ['--audio', 'win', '--au_la', '-1', '--refbase', '--w_lips', '3', '--depth', '6']
ap_only = a.only.split(',') if a.only else []
if a.only: combos = {k: v for k, v in combos.items() if k in a.only.split(',')}
for name, extra in combos.items():
    print('train', name, flush=True)
    run([PY, 'train.py', '--out', f'{SC}/runs/{name}'] + base + ['--amp', a.amp] + extra, f'{SC}/logs/train_{name}.log')
    lg = open(f'{SC}/logs/train_{name}.log').read()
    assert ('DONE 1' if extra[-2:] == ONE else 'DONE 2') in lg and 'nan' not in lg.lower().split('trainable')[-1], name
if 'best_win_refbase' in combos and not os.path.exists(f'{SC}/runs/best_resume/last.pt'):   # resume path (v3_seen recipe: + heldout_ref, boost)
    print('train best_resume (--resume, --splits +heldout_ref, --boost 8)', flush=True)
    run([PY, 'train.py', '--out', f'{SC}/runs/best_resume'] + base[:-6] + ['--splits', f'{TS},heldout_ref', '--boost', '8', '--depth', '6', '--minutes', '120', '--amp', a.amp,
         '--resume', f'{SC}/runs/best_win_refbase/last.pt'] + combos['best_win_refbase'][:-2] + ONE, f'{SC}/logs/train_best_resume.log')
import cv2
clip = next(c['id'] for c in json.load(open(f'{DATA}/manifest.json'))['clips'] if c['split'] == 'heldout' and os.path.exists(f"{SC}/frames/{c['id']}/ok"))
env['CPU_AMP'] = a.amp
for name in [k for k in ('tokens_splat_noise', 's384_tokens_splat_noise', 'none_raster_controlnet', 'best_resume', 'xattn')
             if (k in combos or (k == 'best_resume' and 'best_win_refbase' in combos)) and os.path.exists(f'{SC}/runs/{k}/last.pt')]:
    print('infer', name, flush=True)
    if not os.path.exists(f'{SC}/runs/{name}/out_alips/{clip}.mp4'): run([PY, 'infer.py', '--run', f'{SC}/runs/{name}', '--dev', 'cpu', '--clips', clip, '--max_frames', '8', '--bs', '4', '--modes', 'full,alips']
        + (['--bench'] if name == 'tokens_splat_noise' else []), f'{SC}/logs/infer_{name}.log')
    for sub in ('out', 'out_alips'):
        cap = cv2.VideoCapture(f'{SC}/runs/{name}/{sub}/{clip}.mp4')
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, fr = cap.read()
        assert (n, w, h) == (8, 512, 640) and ok and fr.std() > 1, (name, sub, n, w, h)
        print(f'    {sub}/{clip}.mp4: {n} frames {w}x{h}, pixel std {fr.std():.1f}')
print('SMOKE_OK', 'real' if a.real else 'synthetic', flush=True)
