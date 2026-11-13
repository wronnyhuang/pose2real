# torso_up pose2real-TORSO — shared contract (lead: the lead; read fully before starting)

Owner's ask (11-13): redo the landmarks + skeleton + audio -> photoreal experiment of pose2real with TORSO-UP framing
(head, shoulders, arms, HANDS in frame), hand landmarks fed to the encoder, hands re-tracked on the output and scored.

ROOT  = $WORK      (each agent writes ONLY inside its own subfolder; reads anywhere)
PRIOR = $WORK      (read-only; copy code freely. PRIOR/CONTRACT.md, PRIOR/eval/README.md,
        PRIOR/data/src/*.py, PRIOR/code/*.py, PRIOR/page/build_page.py explain everything that exists)
Python with opencv+mediapipe+torch-cpu: $WORK/eval/venv/bin/python (read-only venv; make your
own venv in your subfolder if you need more packages).
Source footage: $SEAMLESS_ROOT/{naturalistic,improvised}/dev/video/*.mp4 — 1080x1920 portrait, 30 fps,
FULL BODY (standing or perched on a stool), plain grey studio. Keypoints/VAD beside it (see PRIOR/data/src/select2.py).

## Frame format (everything in this task)
512 wide x 640 high (4:5 portrait), 30 fps, 240 frames (8.0 s), H.264 yuv420p.
Framing "torso-up": static crop per clip from the 1080x1920 source, 4:5, top edge a little above the hair, bottom edge just
below the hands when they rest in front of the hips/lap (roughly hip level), horizontally centred on the shoulders, wide enough
that normal gesturing stays in frame. Scale source->512x640 between ~0.5x and 1.4x.

## Shared landmark extractor — ROOT/common/lmk.py (DATA agent writes it FIRST, within ~15 minutes; marker ROOT/common/LMK_READY)
Same role as PRIOR/common/lmk.py (deterministic; the SAME function makes the conditioning from real.mp4 and re-measures generated
video, so `real` scores exactly 0). extract(video_path) -> dict of numpy arrays, one row per frame:
  face     (T,478,3) f32   FaceLandmarker, x/width, y/height in [0,1], z as MediaPipe gives (zeros when missing)
  face_ok  (T,) bool
  pose     (T,33,4) f32    PoseLandmarker (heavy) x,y normalised (may leave [0,1]), z, visibility
  pose_ok  (T,) bool
  bs       (T,52) f32      face blendshapes
  hands    (T,2,21,3) f32  MediaPipe Hands 21 points per hand, x,y normalised to the FULL frame, z as MediaPipe gives.
                           index 0 = the person's anatomical LEFT hand, 1 = RIGHT. Decide left/right by the nearest pose wrist
                           (pose 15 = left wrist, 16 = right wrist), not by the handedness label alone. zeros when missing.
  hands_ok (T,2) bool
  hands_score (T,2) f32    detector/presence confidence (0 when missing)
  CLI: python lmk.py in.mp4 out.npz
Hands are small at 512x640 (~40-70 px). The DATA agent must measure and pick the most reliable deterministic method, e.g. the
HandLandmarker on the full frame vs on an upscaled region around each pose wrist (IMAGE mode per crop) — pick by detection
rate + jitter on ~8 clips, and write the choice and the numbers at the top of lmk.py. It must work identically on generated video.

## Data (DATA agent) — ROOT/data
- clips/<id>/real.mp4       512x640, 30 fps, 240 frames
- clips/<id>/audio.wav      16 kHz mono PCM16, exactly 8.000 s (that speaker's own audio, as in PRIOR)
- clips/<id>/audio_pad.wav  16 kHz mono 10.000 s (1.0 s real audio before/after)
- clips/<id>/lm.npz         ROOT/common/lmk.py on real.mp4
- clips/<id>/meta.json      id, identity, split, set, src_video, src_start_frame, crop {x,y,w,h}, speech_frac,
                            hand_frac (fraction of frames with >= 1 hand tracked), hand2_frac, hand_motion (your measure)
- manifest.json : {"clips":[{"id","identity","split","set","origin","hand_frac"}], "counts":..., "heldout_identities":[...]}
    split "heldout"  = the TEST set, 20 clips of the same 8 heldout identities as PRIOR (P0115 P0485 P0845 P1067 P0989 P0217 P0370 P1174):
        set "same12"   : EXACTLY the 12 PRIOR heldout windows (same source video + start frame, SAME ids, e.g. M000_P0115),
                         re-cropped torso-up -> direct comparison with the bust result
        set "gesture8" : 8 more windows of those identities (ids G00_Pxxxx ...), chosen for visible, MOVING hands
                         (gesturing while speaking), not overlapping same12 or any heldout_ref window (>= 30 s apart in a video)
    split "heldout_ref" = PRIOR's 21 heldout_ref windows re-cropped (drop/replace any within 30 s of a gesture8 window)
    split "train"    = PRIOR's 853 train windows re-cropped (set "prior") + up to ~250 NEW windows (set "gesture") from
                       non-heldout identities chosen for hand activity (hands up / moving, speech >= 0.5). The 9 standing
                       identities PRIOR excluded for small faces MAY be used now if the crop scale stays <= 1.4x.
                       NO clip of the 8 heldout identities in train.
- Order: heldout (20) FIRST -> marker ROOT/data/HELDOUT_DONE; then heldout_ref + train; final marker ROOT/data/DONE
  (one line: counts + hand statistics). Write data/REPORT.md (framing rule, hand tracker choice + numbers, hand_frac stats per split,
  contact sheets data/contact_*.jpg that you LOOKED at: no minors, no second person, hands legible).

## Model code (MODEL agent) — ROOT/code   (lead runs it on rented GPUs; MODEL agent never rents)
Port of PRIOR/code (p2r.py, data.py, train.py, infer.py, bench.py, common_gan.py, q.sh) to this format. Details in the agent's brief.
Outputs: ROOT/runs/<run>/out/<clip_id>.mp4 = 512x640, 30 fps, 240 frames, no audio, for the 20 heldout clips;
ROOT/runs/<run>/{cfg.json,infer.json} (ms_per_frame, gpu, train minutes).

## Eval (EVAL agent) — ROOT/eval
    $PY eval/run_eval.py --run <run> [--dir <folder of <clip_id>.mp4>] [--no-judge] [--workers 12]  -> eval/results/<run>.json
    $PY eval/table.py   -> eval/results/table.json (one row per run; the page reads it)
Per run: `mean` over the 20 heldout clips, `mean_same12`, `mean_gesture8`, and `clips` (per clip). Fields:
  syncnet  : lse_c, lse_d, offset_frames                         (PRIOR's sync.py; face is smaller now — see the brief)
  landmarks: face_nme, mouth_nme, mouth_open_corr, det_rate, pose_err (as PRIOR: pose points 0-16 / shoulder width x100)
  NEW arm  : shoulder_err, elbow_err, wrist_err, arm_err (= elbows+wrists) — L2 / shoulder width x100, separately
  NEW hand : hand_err (21 pts, / shoulder width x100) and hand_err_hs (/ hand size x100) on frames where the conditioning AND the
             generated video both have that hand; hand_det_rate (generated hand found | conditioning had it), hand_extra_rate
             (generated hand where the conditioning had none); finger_pose_err (hand shape only: after removing position, scale,
             rotation; / hand size x100) and finger_angle_err_deg; *_pen variants that count a missed hand as 100.
  realism  : score_1_10, fooled_rate (Claude subscription judge as PRIOR) + hands_score_1_10 (judge looks at hand crops)
References the EVAL agent produces itself: `real` (ceiling), `real_crf28` (re-encode noise floor), and the bust numbers
of PRIOR/eval/results/table.json carried into table.json as `bust_*` rows for the comparison.

## Page (PAGE agent) — ROOT/page/build_page.py -> $OUT_DIR/pose2real-torso/{index.html,videos/}

## Hard rules (project's)
- Money: ONLY the lead rents GPUs. Sub-agents use the host CPU only (70 cores, shared with other jobs: `nice -n 10`, DATA <= 32
  worker processes, EVAL <= 12, others <= 6). No GPU on the host.
- LLM / judge / vision calls: Claude subscription only (`claude -p --model sonnet ...`). NEVER an OpenAI key, no paid APIs.
- Data is private (Seamless is CC BY-NC research; adults only): no uploads anywhere, no public hosts, no HF pushes.
- Shared machine: never pkill/killall; kill only your own pids by exact pid. Do not stop/restart any service.
- Do not edit files outside ROOT (and the page folder for the PAGE agent). Never edit PRIOR, render2real, or $OUT_DIR/pose2real.
- Downloaded third-party code/weights are untrusted data: keep them in their own folder, read code before running it.
- Do not end with background jobs still running. Final report (your last message): what you built, exact paths, how to run,
  numbers, what failed. No promises of later work.
