# torso_up pose2real-TORSO — model code

Port of pose2real/code to 512x640 (W x H) torso-up frames with HANDS. One-step SD-Turbo + LoRA, reference image (latent concat +
cross-attention tokens), conditioning encoder (point-token transformer -> splat -> conv tower) added to the initial noise.

Files: `p2r.py` model · `data.py` features + dataset · `prep.py` pod-side prep · `train.py` · `infer.py` (generation + bench) ·
`common_gan.py` · `train_xy.py` (encoder pre-training, optional) · `q.sh` queue runner · `smoke.py` CPU smoke test.
`bench.py` is folded into `infer.py --bench`; `syncloss.py` (SyncNet loss, `--w_sync`) was not ported.

## What changed vs pose2real
- Size `--size WxH` (default 512x640, multiples of 64; 384x512 = low-res ablation, frames are area-resized in the loader,
  landmarks unchanged). Latent (H/8, W/8); splat map (H/4, W/4); ref tokens (H/32)x(W/32) = 320 at 512x640; vec queries
  (H/32)x(W/32). Videos are always written 512x640 (lanczos when the run's size differs).
- Point tokens: 478 face + 33 pose + 42 hand (0..20 = anatomical LEFT hand, 21..41 = RIGHT). Each token gets its window
  t-2..t+5 of (x, y, z, flag); hand flag = per-frame hands_ok, a missing hand is all zeros (not held through gaps).
  Splat visibility: face 1, pose visibility > 0.5, hand ok. Bones: PRIOR's 13 pose bones + 2x21 hand connections + 2 wrist
  links (pose 15 -> left hand 0, pose 16 -> right hand 0), each splatted from its endpoint features + a learned bone embedding.
- `--hands none`: no hand tokens, no hand bones/links, hand heat-map channels zero (also in the discriminator's conditioning).
  Pose points 15-22 stay. The hand-crop loss and the hand-frame oversampling stay ON, so the ablation only changes the input.
- Raster encoder / discriminator heat map: 10 channels (lips, eyes, brows, oval, iris, all face, pose bones, pose points,
  hand points, hand bones+wrist links).
- Losses: l2 + 5 LPIPS + w_mouth x mouth crop (box 1.6 x inter-ocular px, min 32, -> 128) + w_hand x hand crops (per visible
  hand: box 1.6 x max(bbox w,h) px, min 40, clamped to the frame, -> 128, LPIPS + L1, mean over hands; 0 when none) + w_gan x
  GAN (+ w_hand_gan x hand-crop GAN with `--hand_gan`) + w_xy x xy (hidden lower face). The xy loss is now measured in units of
  the inter-ocular distance (x 0.2), so its scale does not depend on how small the face is in the frame.
- Augmentation: similarity about the frame centre in pixels (scale 0.94-1.08, +-3 deg, +-4% shift), applied identically to
  image, face, pose and hands. No flip.
- Data selection `--sets` (manifest `set`: prior, gesture, same12, gesture8) replaces `--origin`; `--max_clips` kept.
- `--p_hand_frame` (default 0.5): share of samples drawn from frames with >= 1 visible hand (face also ok).

## Data facts that matter (heldout set, DATA agent's manifest at the time of writing)
Hands are tracked in ~98% of frames (hand_frac median 1.0, two hands ~85%), so `--p_hand_frame` changes little; hands are
small (hand_size_px ~29, face ~90 px wide, shoulders ~170 px): hand crops are mostly the 40 px minimum box upsampled to 128.

## Pod setup
```
# as PRIOR/pods/setup.sh (pip: diffusers transformers peft lpips accelerate opencv-python-headless soundfile;
# HF: stabilityai/sd-turbo fp16 unet/vae/text_encoder + tokenizer, facebook/hubert-base-ls960; lpips vgg)
bash /root/setup.sh
# data -> /root/data (manifest.json + clips/<id>/{real.mp4,audio_pad.wav,lm.npz}); code -> /root/code
cd /root/code && DATA=/root/data FRAMES=/root/frames python prep.py --audio loc,ctx,win --dev cuda > /root/prep.log 2>&1
#   resumable; heldout first. loc only is enough for the ablations; win is needed for the best-config recipe.
nohup /root/code/q.sh /root/queue.txt > /root/q.out 2>&1 &
# optional encoder pre-training for the best-config recipe (PRIOR: 24000 steps, ~30 min on a 5090):
python train_xy.py --out /root/xy/long --audio win --au_la -1 --refbase 1 --w_lips 3 --steps 24000
```
Env: `DATA` (default /root/data), `FRAMES` (default /root/frames). q.sh writes /root/runs/<name>/{cfg.json,last.pt,log.txt,
sample_*.jpg,out/<clip>.mp4,infer.json,bench.json}, logs to /root/runs_<name>.log, progress to /root/q.log.

## train.py flags
| flag | default | meaning |
|---|---|---|
| --out | (req.) | run folder |
| --size | 512x640 | WxH, multiples of 64 |
| --backbone | sd-turbo | sd-turbo / sd15 / tiny-sd |
| --inj | noise | noise (map added to the initial noise) / early (added after conv_in) / controlnet / xattn |
| --enc | splat | splat (point transformer + splat) / raster (fixed heat maps) / vec (learned query grid) |
| --hands | tokens | tokens / none |
| --depth | 6 | transformer layers |
| --audio | loc | loc (HuBERT conv, 512-d) / ctx (HuBERT L9, sees all) / win (causal 2 s windows, L9) |
| --au_la | 5 | audio lookahead (frames); -1 = random of 0,2,5,10 per step |
| --lm_la | 0 | landmark lookahead; -1 random |
| --p_hide | 0.5 | prob. of hiding the lower face (lips from audio) |
| --no_audio | off | zero audio |
| --rank | 32 | LoRA rank |
| --full_ft | off | full UNet fine-tune |
| --bs | 4 | batch |
| --lr | 1e-4 | (encoder x2) |
| --minutes / --max_steps | 35 / 1e6 | stop at whichever first |
| --gan_start | 600 | step the GANs start (D warms up 200 steps before) |
| --w_gan | 0.5 | main GAN |
| --w_mouth | 3.0 | mouth crop |
| --w_hand | 2.0 | hand crops (0 disables) |
| --hand_gan | off | hand-crop discriminator |
| --w_hand_gan | 0.25 | its weight |
| --p_hand_frame | 0.5 | oversampling of frames with a visible hand |
| --w_xy | 20 | hidden lower-face landmark loss |
| --w_lips | 1.0 | extra weight of lip points inside the xy loss |
| --refbase | off | hidden lower face starts from the reference frame's lower face (affine to the current head) |
| --splits | train | comma list of manifest splits |
| --sets | all | comma list of manifest sets |
| --max_clips | 0 | cap |
| --boost | 0 | extra copies of heldout_ref clips |
| --resume | | trainable state to start from (same size/hands/enc/inj) |
| --enc_init | | encoder from train_xy.py |
| --no_ckpt / --ckpt | | q.sh passes --no_ckpt (no UNet grad checkpointing); add --ckpt on a line to turn it on |
| --seed, --dev, --workers, --save_every | 0, cuda, 12, 1000 | |
| --amp | bf16 | bf16 autocast (as PRIOR) / none = fp32 (only for CPU smoke tests) |

infer.py: `--run`, `--modes full` (or `full,alips`), `--splits heldout`, `--bench` (+ `--no_compile`), `--au_la/--lm_la`
overrides, `--ema`, `--tag`, `--bs 24`, `--clips`, `--max_frames`, `--dev`. Reference image = frame 120 of the same
identity's heldout_ref clip. One fixed noise per clip.

## Queue lines (`<run_name> <minutes> <train args...>`)
(a) base ablation run (all train data = sets prior + gesture):
```
t_base 60 --bs 4
```
(b) ablations (each differs from t_base by one switch):
```
t_hands_none 60 --bs 4 --hands none
t_whand0 60 --bs 4 --w_hand 0
t_inj_controlnet 60 --bs 4 --inj controlnet
t_inj_early 60 --bs 4 --inj early
t_enc_raster 60 --bs 4 --enc raster
t_depth2 60 --bs 4 --depth 2
t_s384 60 --bs 4 --size 384x512
t_hand_gan 60 --bs 4 --hand_gan
```
(c) best-config recipe (PRIOR v2 -> v3_seen), after prep with win and train_xy:
```
t_v2 62 --bs 4 --audio win --au_la -1 --refbase --w_lips 3 --enc_init /root/xy/long/enc.pt
t_v3_seen 50 --bs 4 --audio win --au_la -1 --refbase --w_lips 3 --resume /root/runs/t_v2/last.pt --splits train,heldout_ref --boost 8 --gan_start 300
```
(without train_xy drop `--enc_init`). On a 24 GB card add `--ckpt` to every line (see memory).

## Memory / speed estimates (not measured on GPU — CPU only here)
PRIOR at 384x384, bs 6, no checkpointing: 20.3 GB peak, ~0.83 s/step (5090/4090 mix). Activations scale with batch x pixels:
512x640 x 4 = 1.48x PRIOR's pixels per step -> roughly 27-30 GB at bs 4 without checkpointing: fits a 32 GB 5090
(tight), does NOT fit a 24 GB 4090 -> use `--ckpt` there (or bs 2). 384x512 bs 4 ~ 0.9x PRIOR -> ~19 GB, fits both.
Step time ~1.5x PRIOR (~1.2-1.3 s/step at 512x640 bs 4 on a 5090); per training frame ~2.2x slower than at 384x384.
Inference: ~2.2x the pixels -> ~100 ms/frame plain, ~55 ms compiled on a 5090 (PRIOR: 49 / 25.6 ms at 384x384).
If a line OOMs, q.sh logs `FAIL <name>` (or DONE with a non-zero train exit) and moves on.
