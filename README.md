# pose2real — face landmarks + body skeleton + audio → photoreal talking video

No render in the loop. A trainable encoder (point-token transformer over 478 face + 33 pose landmarks, plus HuBERT audio tokens) is
mixed into one map that is **added to the starting noise of one-step SD-Turbo** (LoRA r32, GAN + LPIPS + mouth-crop losses). A reference
photo of the person fixes the look. 384×384, 30 fps, 8-second clips. `torso_up/` ports it to 512×640 torso-up frames with hands (in progress).

## Results (12 held-out clips, 8 people; `eval/results/table.json`)
| run | LSE-C ↑ | LSE-D ↓ | face err % | mouth err % | skeleton err % | mouth-open corr | realism /10 |
|---|---|---|---|---|---|---|---|
| real video (ceiling) | 6.16 | 7.83 | 0 | 0 | 0 | 1.00 | 8.08 |
| real, H.264 crf28 (noise floor) | 5.99 | 7.92 | 0.46 | 0.50 | 0.69 | 0.99 | – |
| render→photoreal fast (sibling project) | 3.11 | 9.94 | 2.07 | 2.03 | 2.45 | 0.93 | 5.42 |
| `base` (126 clips, ablation reference) | 2.41 | 11.09 | 2.36 | 2.31 | 2.43 | 0.83 | 5.08 |
| `v2` (853 clips, pretrained encoder) | 3.49 | 9.86 | 2.11 | 2.02 | 2.25 | 0.90 | 5.33 |
| `v4_more` (unseen people) | 3.41 | 9.94 | 1.75 | 1.88 | 2.30 | 0.93 | 5.46 |
| **`v3_seen`** (16–32 s of each test person in training) | 3.47 | 9.84 | **1.64** | **1.73** | **1.66** | 0.93 | 5.17 |
| `v5_sync` (SyncNet loss — reward-hacks: eye sparkle) | 4.25 | 9.61 | 2.07 | 2.05 | 1.75 | 0.91 | 3.12 |
| `v3_seen` audio-only lips (lower-face landmarks hidden) | 1.29 | 11.89 | 1.87 | 2.48 | 1.67 | 0.61 | – |

Ablations (all in the table file): early-layer / cross-attention-only / ControlNet-style injection, encoder depth 2/6/12, raster vs deep
encoder, SD1.5 and Tiny-SD backbones. Findings: noise injection with the splat encoder wins; lip sync (SyncNet) is the open problem
(3.5 vs 6.2 real); realism 5.2 vs 8.1 real; speed **25.6 ms/frame (39 fps)**, RTX 5090 with fused LoRA + `torch.compile`.
Details: `page/{summary,worked,didnt,next}.md`, `eval/README.md` (metric definitions + SyncNet validation), `CONTRACT.md` (data/feature formats).
Per-run training arguments: `run_configs/*.json`. GPU spend $11.33 (RunPod).

### torso_up (512×640, hands) — in progress
`torso_up/` contains code, run configs, eval scripts and results so far (`torso_up/eval/results/table.json`; the work was still running when this
repo was exported — numbers there are interim and lower than the bust-crop model; see `torso_up/code/README.md`).

## Hardware
1× RTX 5090 32 GB for training and benchmarks (50–62 min per full run), 1× RTX A5000 24 GB for queue runs; CPU box (70 cores) for data prep (MediaPipe) and the eval harness. Python 3.12, torch 2.8.0+cu128, diffusers 0.43.0, peft 0.21.2.

## Base models
`stabilityai/sd-turbo`, `facebook/hubert-base-ls960`; MediaPipe `face_landmarker` and `pose_landmarker_heavy` task files; eval uses
[joonson/syncnet_python](https://github.com/joonson/syncnet_python) (`syncnet_v2.model`, `sfd_face.pth`).

## Reproduce, stage by stage
```bash
. scripts/env.sh
bash scripts/00_setup.sh
bash scripts/01_fetch_artifacts.sh <DRIVE_FOLDER>   # lm.npz landmarks + meta, weights, eval outputs (ARTIFACTS.md)
bash scripts/02_data.sh --media-only                # re-cut real.mp4 + audio from your own Seamless Interaction copy (DATA.md)
#   or full rebuild: bash scripts/02_data.sh         # (needs ../render2real data for the 138 seed clips: R2R_DATA)
bash scripts/03_train.sh                            # encoder warm start -> v2 -> v4_more / v3_seen
bash scripts/04_infer.sh $RUNS/v3_seen
bash scripts/05_eval.sh v3_seen
```
Layout: `code/` (model `p2r.py`, `data.py`, `train.py`, `train_xy.py`, `infer.py`, `syncloss.py`), `common/lmk.py` (shared deterministic landmark
extractor — it both makes the conditioning and re-measures generated video), `data/src/` (selection, cut, audio, QC), `eval/`, `page/`, `run_configs/`, `torso_up/`.
Real videos and voices are not redistributed; see `DATA.md`.
