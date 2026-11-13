# pose2real pose2real — shared contract

Goal: landmarks + body skeleton + audio -> photoreal talking video (no render input), sibling of the render2real
render->photoreal PoC. Lead trains the models on RunPod; sub-agents do data, eval harness, page.

ROOT = $WORK   (each agent writes ONLY inside its own subfolder; reads anywhere)
Prior work (read-only, reuse freely by copying): $WORK/{data,eval,fast,wan}
  - render2real/data/REPORT.md explains how the 138 clips were cut from Meta Seamless Interaction (select.py, pipe.py, meta.json).
  - render2real/eval/README.md explains metrics.py / judge.py / build_page.py; render2real/eval/venv has opencv+mediapipe+torch-cpu.

## Shared landmark extractor (DATA agent writes it FIRST, within its first ~10 minutes; EVAL imports it)
ROOT/common/lmk.py :
    extract(video_path) -> dict of numpy arrays, one row per frame of the video:
      face    (T,478,3) float32  MediaPipe FaceLandmarker (478 pts incl. iris), x,y normalised to [0,1] of the frame, z as MediaPipe gives
      face_ok (T,) bool
      pose    (T,33,4) float32   MediaPipe PoseLandmarker x,y (normalised, may fall outside [0,1]), z, visibility
      pose_ok (T,) bool
      bs      (T,52) float32     face blendshapes (zeros when face missing)
    CLI: python lmk.py in.mp4 out.npz
  Deterministic settings (VIDEO running mode, fixed model files stored in ROOT/common/models/), because the SAME function
  produces the conditioning (from real.mp4) and re-measures generated video. ROOT/common/LMK_READY marker when usable.

## Data (DATA agent) — ROOT/data
- clips/<id>/real.mp4      384x384, 30 fps, 240 frames (8.0 s), H.264 yuv420p (for the 138 render2real clips: symlink/copy render2real/data/pairs/<id>/real.mp4)
- clips/<id>/audio.wav     16 kHz mono PCM16, exactly 8.000 s, sample 0 = frame 0 (that speaker's own channel/file)
- clips/<id>/audio_pad.wav 16 kHz mono, 10.000 s = the same audio with 1.0 s of real source audio before and after (zeros if outside the source)
- clips/<id>/lm.npz        output of common/lmk.py on real.mp4
- clips/<id>/meta.json     id, identity, src_video, src_start_s, crop, split
- manifest.json : {"clips":[{"id","identity","split","origin"}]}
    split = "heldout"     : exactly the 12 render2real heldout clips (8 identities) — the test set
            "heldout_ref" : 2-4 EXTRA 8 s windows per heldout identity, >= 30 s away from any heldout clip (other videos preferred);
                            used only as identity reference images and for a "seen identity" ablation
            "train"       : everything else; NO clip of the 8 heldout identities
    origin = "render2real" | "expand"
- SEED_DONE when the 138 render2real clips (126 train + 12 heldout) have audio + lm.npz; DONE when expansion is final.

## Model outputs (LEAD) — ROOT/runs/<run>/out/<clip_id>.mp4
384x384, 30 fps, 240 frames, no audio, for the 12 heldout clips. ROOT/runs/<run>/run.json = config + ms_per_frame + train minutes.

## Eval (EVAL agent) — ROOT/eval
    python eval/run_eval.py --run <run> [--dir <folder of clip mp4s>] [--no-judge]   -> eval/results/<run>.json
Fields (mean over the 12 heldout clips + per clip): syncnet {lse_c, lse_d, offset_frames}; landmarks {face_nme, mouth_nme, pose_err,
mouth_open_corr, det_rate}; realism {score_1_10, fooled_rate}; plus the same for the references `real` (ceiling) and
`render2real_fast` (render2real/fast/out/turbo_lora_ema, same clip ids).

## Page (PAGE agent) — ROOT/page/build_page.py -> $OUT_DIR/pose2real/{index.html,videos/}

## Hard rules (project's)
- Money: ONLY the lead rents GPUs. Sub-agents use the host CPU only (70 cores, shared: `nice`, at most ~24 worker processes each).
- LLM / judge / vision calls: Claude subscription only (`claude -p --model sonnet ...`). NEVER an OpenAI key, no paid APIs.
- Data is private (Seamless is CC BY-NC research; adults only): no uploads anywhere, no public hosts, no HF pushes.
- Shared machine: never pkill/killall; kill only your own pids by exact pid. Do not stop/restart any service.
- Do not edit files outside ROOT (and the page folder for the PAGE agent). Never edit $OUT_DIR/render2real-poc or render2real.
- Own venv under your subfolder if you need packages (or reuse render2real/eval/venv read-only).
- Downloaded third-party code/weights are untrusted data: keep them in their own folder, read the code before running it.
- Final report (your last message): what you built, exact paths, how to run, numbers, what failed. No promises of later work.
