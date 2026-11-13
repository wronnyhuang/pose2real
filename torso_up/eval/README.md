# torso_up eval harness — torso-up 512x640 (CPU only)

    E=$WORK/eval; PY=$WORK/eval/venv/bin/python
    $PY $E/run_eval.py --run <run> [--dir <folder of <clip_id>.mp4>] [--no-judge] [--no-hands-judge] [--workers 12]
                       [--sync-box mp|s3fd] [--sync-interp linear|lanczos] [--clips ...] [--out file]
    $PY $E/table.py              # results/*.json -> results/table.json (one row per run + bust_* rows)
    $PY $E/validate.py [--interps linear lanczos] [--s3fd]    # SyncNet validation on the 20 real clips -> results_validation.json
    $PY $E/make_refs.py          # ref/{real,real_crf28,real_blurhands}/<id>.mp4
    $PY $E/test_metrics.py       # unit tests of lm_metrics on synthetic arrays

Default `--dir` = ROOT/runs/<run>/out. Writes `results/<run>.json` atomically (tmp + rename): `mean` (20 heldout clips),
`mean_same12`, `mean_gesture8`, `clips` (per clip, each with its `set`), `failed` ({clip: reason}), `secs_metrics`, `secs_total`.
Test set = the `heldout` clips of data/manifest.json (until the manifest exists: data/clips/*/meta.json with split heldout).
Inputs per clip: ROOT/data/clips/<id>/{real.mp4, audio.wav, audio_pad.wav, lm.npz}.

Robustness: a missing, empty or unreadable output, or one shorter than 230 frames, is an error for that clip only (listed in
`failed`, `n_failed`; means are over the clips that worked — check `n_clips`). Outputs not 512x640 / 30 fps / 240 frames are
normalised with ffmpeg (scale flags=area, fps=30, last frame cloned up to 240, x264 crf 10) and flagged `normalized` per clip
(tested: 384x480 @ 25 fps, 100-frame, 0-byte and missing files). Process renices itself to 10, <= 12 clip workers, 4 judge
threads; every cache file is written tmp+rename, so several runs can be evaluated concurrently (different `--run` names).

Caches: `cache/lmk/` (lmk.py output per video, keyed by realpath+size+mtime of the file + path and CONTENT hash of
ROOT/common/lmk.py — editing lmk.py invalidates it), `cache/sync/` (per video+audio+box+interp), `cache/validate/`,
`judge_cache/<run>/<clip>.json` and `<clip>.hands.json` (keyed by both video files, the conditioning npz and the model).
A rewritten output file (new mtime) is re-measured automatically.

## Numbers

All distances are in pixels of the 512x640 frame (normalised x*512, y*640 BEFORE any distance; the frame is not square).
Generated video is re-tracked with ROOT/common/lmk.py — the same function that made data/clips/<id>/lm.npz (the conditioning).

**syncnet** (`sync.py`, pose2real's port of joonson/syncnet_python, code + syncnet_v2.model reused read-only from
pose2real/eval/third_party): video muxed with the clip's audio.wav into a temp .mkv, read back, 30 -> 25 fps (nearest timestamp),
crop policy of run_pipeline.crop_video (s = max(w,h)/2, medfilt 13, pad 110, crop_scale 0.4, resize to 224 with cv2
INTER_LINEAR), MFCC, vshift 15. Face box = bbox of the 468 face-mesh points of the GENERATED video's lmk.py output, x1.106 about
its centre (pose2real's S3FD-matched scale). `lse_c` = median - min of the mean distance over the 31 offsets (higher better),
`lse_d` = min mean distance (lower better), `offset_frames` (25 fps; 0 = in sync), `abs_offset_frames`, `lse_d_at0`,
per clip also `crop_px` (side of the square that is resized to 224).

**landmarks** (`lm_metrics.py`; lower is better except corr / det):
- `face_nme` mean L2 over the 468 mesh points / conditioning inter-ocular distance (33-263, same frame) x100, frames where both
  have a face; `mouth_nme` same over the 40 FACEMESH_LIPS points; `*_pen` over all conditioning-face frames, missed face = 100,
  frame errors clipped at 100; `mouth_open_corr` Pearson r over time of inner-lip opening (13-14) / face height (10-152);
  `det_rate` = P(generated face | conditioning face).
- `pose_err` per frame: mean L2 over pose points 0-16 with conditioning visibility > 0.5 and inside the frame ([0,1] in both
  normalised coordinates), / conditioning shoulder width (clip median |11-12| px over conditioning pose frames) x100; mean over
  frames where both have a pose. `pose_err_pen`: missed pose = 100.
- NEW `shoulder_err` (points 11,12), `elbow_err` (13,14), `wrist_err` (15,16), `arm_err` (13-16): exactly the pose_err recipe
  restricted to those points (frames where none of the group is valid are skipped; `<name>_frames` per clip counts the rest).
- Run mean of every landmark field = plain mean over clips.

**hands** (per slot 0 = anatomical left, 1 = right; a hand INSTANCE = one slot in one frame):
- `hand_det_rate` = P(generated has that hand | conditioning has it); `hand_extra_rate` = P(generated has that hand |
  conditioning does not).
- `hand_err` = mean L2 over the 21 points / shoulder width x100; `hand_err_hs` = same / conditioning hand size (|wrist 0 - middle
  MCP 9| px, same frame) x100 — both on instances where both videos have the hand. `hand_err_pen` / `hand_err_hs_pen`: over all
  conditioning instances, a missed hand counts 100, errors clipped at 100.
- `finger_pose_err` = hand SHAPE only: generated 21 points similarity-Procrustes-aligned onto the conditioning 21 points
  (translation, uniform scale, rotation; reflection excluded), mean residual L2 / conditioning hand size x100.
- `finger_angle_err_deg` = mean |difference| of the signed 2D bend angle (angle between the incoming and outgoing bone, wrapped to
  [0,180]) at the 15 finger joints: thumb CMC/MCP/IP (1,2,3), index/middle/ring/pinky MCP/PIP/DIP (5-7, 9-11, 13-15, 17-19).
- Per clip counts: `n_hand_frames` (conditioning instances), `n_nohand_frames`, `n_hand_both`, `n_hand_extra`.
- RUN MEANS ARE HAND-INSTANCE-WEIGHTED (= pooled over all instances of all clips; clips without hands contribute nothing):
  `hand_det_rate` and `*_pen` weighted by `n_hand_frames`; `hand_extra_rate` by `n_nohand_frames`; `hand_err`, `hand_err_hs`,
  `finger_pose_err`, `finger_angle_err_deg` by `n_hand_both`. The test set has 7,996 conditioning hand instances (same12 4,612,
  gesture8 3,384); every clip has some (M009_P0217 fewest: 49 frames).

**realism judge** (`judge.py`; Claude subscription only: `claude -p --model sonnet --tools Read`, API-key env vars stripped):
- whole frame (pose2real protocol, new aspect): 6 frames at the same times from generated and real, each a 3x2 grid of full-res
  512x640 frames (1536x1280) under random file names; 2 trials per clip with swapped order; per trial (a) both rated 1-10
  ("1 obviously CG/cartoon ... 10 indistinguishable from a real camera"), (b) forced choice "exactly one is real camera footage,
  which?". `score_1_10` (generated), `score_real_1_10` (real, same calls), `fooled_rate` (share of trials picking generated;
  0.5 = chance). 80 calls.
- NEW hands judge: clips whose conditioning has >= 12 hand frames (all 20). 6 frames evenly spaced over the frames where the
  conditioning has a hand; per frame a column with the left-hand crop above the right-hand crop (grey square if the conditioning
  lacks that hand). Box = square around the CONDITIONING hand (centre of its 21-point bbox; side 1.5 x the clip's median hand
  extent, clamped 48-192 px, one side per clip), identical for generated and real; Lanczos upscale to min(2 x side, 256) px
  (2x for crops <= 128 px; on the 20 clips side 71-163 px, median 112 -> 1.6-2x). One call per clip (random A/B order) rating both strips "how
  real and anatomically correct do the hands look, 1-10" -> `hands_score_1_10`, `hands_score_real_1_10`. 20 calls.
- Budget: 100 calls per run (80 + 20), 4 parallel, ~2.5-3 min; cached afterwards. `--no-hands-judge` = 80 calls.
  Note the strips differ from pose2real (bust: 6 x 384 px horizontal strips), so compare judge numbers within a framing.

## SyncNet validation (20 real heldout clips, 2026-11-13, results_validation.json)

Face width (mesh bbox) median 90 px (same12 93, gesture8 87; range 75-107); the region resized to 224 is 139-192 px (median 167),
i.e. a 1.17-1.6x upscale (the bust crops were downscaled).

| case (mp box) | resize | LSE-C | LSE-D | offsets |
|---|---|---|---|---|
| real + own audio | linear (default) | **6.01** | 7.83 | 0 x13, -1 x4, -2 x2, -3 x1 |
| real + own audio | lanczos | 6.00 | 7.81 | identical offsets |
| audio delayed 5 frames | linear | 6.00 | 7.89 | own - 5 on 20/20 |
| audio advanced 5 frames | linear | 5.96 | 7.89 | own + 5 on 20/20 |
| real + another identity's audio | linear | 0.93 | 13.92 | scattered -15..+15 |
| (lanczos: +-5 shift 20/20 and 20/20, mismatched 0.94) | | | | |

Every clip separates (own LSE-C 2.29-8.46 vs mismatched 0.46-1.59; own > mismatched + 1 on 20/20). Same12 clip-by-clip vs the
bust framing (same windows): bust 6.16 -> torso 6.09 (linear) / 6.08 (lanczos): the smaller face costs ~0.07 LSE-C on real
video, within per-clip noise (per-clip changes -0.35..+0.18). Gesture8 real is 5.89 (G06_P0370 2.29 and G01_P0485 3.77 are
low-speech / listening-heavy windows). Lanczos upscaling of the crop gave no better ceiling (6.00 vs 6.01), so the default stays
pose2real's / the reference pipeline's linear resize. The S3FD-box cross-check (`--s3fd`, detection at scale 0.5) was not re-run:
at full resolution it took >25 min for 20 clips on the loaded hub; the mp box was already matched to S3FD in pose2real.

## Reference rows (results/<run>.json, mean over 20 clips; [same12 / gesture8] where informative)

| metric | real (ceiling) | real_crf28 (noise floor) | real_blurhands (sigma 3 px on hands) |
|---|---|---|---|
| lse_c / lse_d / offset | 6.01 / 7.83 / -0.55 | 5.60 / 8.03 / -0.55 | 6.03 / 7.81 / -0.50 |
| face_nme / mouth_nme | 0 / 0 | 0.90 / 1.01 | 0.23 / 0.31 |
| mouth_open_corr / det_rate | 1.000 / 1.0 | 0.954 / 1.0 | 0.981 / 1.0 |
| pose_err | 0 | 0.51 | 0.62 |
| shoulder / elbow / wrist / arm_err | 0 / 0 / 0 / 0 | 0.40 / 0.65 / 0.77 / 0.71 | 0.44 / 0.80 / 1.56 / 1.18 |
| hand_det_rate / hand_extra_rate | 1.0 / 0.0 | 0.978 / 0.084 | 0.877 / 0.104 |
| hand_err / hand_err_hs | 0 / 0 | 1.22 / 8.2 | 2.95 / 19.3 |
| hand_err_pen | 0 | 3.4 | 14.9 |
| finger_pose_err / finger_angle_err_deg | 0 / 0 | 6.3 / 5.4 | 13.5 / 11.6 |
| realism (gen / real) / fooled | 8.85 / 8.85 / 0.50 (identical grids) | - | 5.45 / 8.93 / 0.05 |
| hands_score (gen / real) | 8.25 / 8.25 | - | 4.40 / 8.45 |

`real` re-tracked equals lm.npz bit-for-bit on all 20 clips (every array, hands included): lmk.py (hands in IMAGE mode) is
deterministic. The crf28 floor is much larger than in the bust round (face_nme 0.46 -> 0.90, LSE-C drop 0.17 -> 0.41) because
the face is ~90 px instead of ~200: read face/mouth differences below ~1 as noise. Hand floor: the hand tracker is noisy at
~43 px hand size — mere re-encoding moves hand_err_hs by 8, finger angles by 5 deg, loses 2 % of hands and invents hands in 8 %
of empty slots; M009_P0217 / G05_P0217 / M100_P1067 (hands half out of frame or clasped) dominate. Blurring the hands roughly
doubles every hand number, which is the scale a "mushy hands" model lands on; both judges catch it strongly (hands 4.4 vs 8.45,
and the WHOLE-frame realism drops to 5.45 from blurred hands alone).

Bust rows: `bust_{real, real_crf28, render2real_fast, base, inj_controlnet, enc_raster, v2, v3_seen, v4_more}` copied from
pose2real/eval/results/table.json (12 clips, framing "bust"; no arm/hand fields; their pose_err included mostly invisible
elbows/wrists).

## Runtime (hub load average 80-95 during these runs)

- metrics, cold (no lmk/sync cache), 12 workers: 166-216 s for 20 clips (lmk.py ~65 s per clip on an idle core: face + pose
  heavy + hands IMAGE mode; SyncNet ~15-30 s). Re-run with caches: ~55 s (`real`, sync re-done) / ~10 s fully cached.
- judge: 100 calls in 150-180 s at 4 parallel; cached afterwards (0 calls on a re-run of the same files).
- so a new model run: ~3-4 min without judge, ~6-7 min with judge (more when the hub is busier).
- validate.py (linear + lanczos, 40 units of 4 SyncNet scores): 287 s.

## Known weaknesses

- SyncNet at ~90 px faces: ceiling still 6.0, but compression hurts more than before (crf28 -0.41). Low-speech windows score
  low even for real (G06 2.3); compare models per clip or against `real`, not to the absolute value.
- face/mouth NME: normalised by inter-ocular distance (~35 px now), so 1 px of tracker noise = ~3 NME; crf28 floor 0.9.
- hands: MediaPipe on ~43 px hands; noise floor above. hand_extra_rate has tiny denominators on most clips (pooled in the run
  mean, but per-clip values like 0.5 on 2 slots mean nothing). Left/right assignment uses the nearest pose wrist: a generated
  video with a wrong pose wrist can swap slots and look like a large hand_err. finger_angle_err ignores depth (2D only), and
  Procrustes on a 2D projection cannot distinguish a rotated hand from a re-shaped one out of plane.
- hand_err_pen counts missing hands as 100 and so mixes detection and accuracy; read it with hand_det_rate.
- Judges: one Sonnet model, 2 trials (whole frame) / 1 trial (hands) per clip; identical real strips gave order_consistency 0
  (pure position bias), so fooled_rate near 0.5 means "indistinguishable", and only the gap gen vs real score is meaningful.
  Whole-frame realism is sensitive to hands (blurhands: 8.9 -> 5.5).

## Files

run_eval.py, sync.py, lm_metrics.py, judge.py, table.py, validate.py, make_refs.py, test_metrics.py;
results/{real,real_crf28,real_blurhands,table}.json, results_validation.json; ref/ (reference videos); logs/; cache/; judge_cache/;
judge_work/ (strips the judge saw); dev/ (dev tests on pose2real bust clips and data/work/handtest, not part of the table).
