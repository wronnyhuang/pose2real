# pose2real eval harness (CPU only)

    E=$WORK/eval; PY=$WORK/eval/venv/bin/python
    $PY $E/run_eval.py --run <run> [--dir <folder of <clip_id>.mp4>] [--no-judge] [--workers 12] [--sync-box mp|s3fd] [--clips ...]
    $PY $E/table.py            # merge results/*.json -> results/table.json (one row per run; the page reads this)
    $PY $E/validate.py         # SyncNet validation on the real clips -> results_validation.json

Default dir = ROOT/runs/<run>/out. Writes `results/<run>.json` atomically: `mean` (over the 12 heldout clips) + `clips` (per clip).
Test set = the 12 `heldout` ids of the manifest. Inputs: ROOT/data/clips/<id>/{real.mp4,audio.wav,audio_pad.wav,lm.npz} once
ROOT/data/SEED_DONE exists (before that: render2real pairs + audio cut into eval/scratch/audio). Videos that are not 384x384 / 30 fps /
240 frames are normalised with ffmpeg first (noted per clip as `normalized`). The process renices itself to 10; 12 clip workers.

Caches (re-runs are cheap): `cache/lmk/` (common/lmk.py output per video file, keyed by realpath+size+mtime+lmk.py signature),
`cache/sync/` (SyncNet result per video+audio+box mode), `judge_cache/<run>/<clip>.json` (judge, keyed by both files + model).
A rewritten output file (new mtime) is re-measured automatically.

## Numbers
**syncnet** (`sync.py`; joonson/syncnet_python code + syncnet_v2.model in third_party/, CPU): video muxed with the clip's
audio.wav into a temp .mkv (video copied, PCM audio) and read back; frames 30 -> 25 fps (nearest timestamp, like `ffmpeg -r 25`);
crop policy of run_pipeline.crop_video (s = max(w,h)/2, medfilt 13, pad 110, crop_scale 0.4, 224 px); MFCC
(python_speech_features, 16 kHz); vshift 15, batch 20, same distance/confidence math as SyncNetInstance.evaluate.
`lse_c` = median - min of the mean distance over shifts (higher better), `lse_d` = min mean distance (lower better),
`offset_frames` = AV offset at 25 fps (0 = in sync; negative = audio lags the lips), `abs_offset_frames`, `lse_d_at0` = distance at
offset 0. Face box: default `mp` = bbox of the 468 face-mesh points from common/lmk.py (computed anyway for the landmark metrics),
scaled x1.106 about its centre so its size matches S3FD's box (median ratio after scaling: 0.999); `--sync-box s3fd` uses the
original S3FD detector (conf 0.9, scale 0.25) instead. Single face track; frames with no box are interpolated.

**landmarks** (`lm_metrics.py`; generated video re-measured with ROOT/common/lmk.py — the same function that made lm.npz):
`face_nme` = mean L2 over the 468 mesh points (px) / conditioning inter-ocular distance (33-263, same frame) x100, frames where both
have a face; `mouth_nme` = same over the 40 FACEMESH_LIPS points; `mouth_open_corr` = Pearson r over time of inner-lip opening
(13-14) / face height (10-152); `pose_err` = mean L2 over pose points 0-16 that have conditioning visibility > 0.5 and lie inside the
frame / conditioning shoulder width (clip median 11-12) x100; `det_rate` = fraction of frames with a generated face.
`*_pen` variants count every frame: a missed face/pose = 100 (one IOD / shoulder width), frame errors clipped at 100. Lower is better
except corr and det_rate.

**realism** (`judge.py`; Claude subscription only: `claude -p --model sonnet --tools Read`, same call pattern as render2real): 6-frame
strips of generated and real (same frame times) under random names; 2 trials per clip with swapped order; per trial (a) both rated
1-10 with render2real's realism wording, (b) forced choice "exactly one is real camera footage, which one?". `score_1_10` = mean realism of
the generated strip, `score_real_1_10` = the real strip in the same calls, `fooled_rate` = fraction of trials where the judge picked
the generated strip (0.5 = indistinguishable). 48 calls per run, 4 parallel. Caveat: render2real's ~8.0 for the fast model was rated next
to the CG render; here the partner is the real video, so the same model scores 5.4 — compare within this table only.

## Validation (12 heldout real clips, official ROOT/data, 2026-11-13)
| case | LSE-C | LSE-D | offsets (frames) |
|---|---|---|---|
| real + own audio (mp box) | 6.16 | 7.83 | 0 x8, -1 x3, -3 x1 (M076) |
| real + own audio (S3FD box) | 6.18 | 7.85 | same as mp except M075 0 |
| audio delayed 5 frames | 6.16 | 7.91 | -5 x9, -6 x2, -8 x1 (= own - 5, except M075) |
| audio advanced 5 frames | 6.12 | 7.90 | +5 x8, +4 x3, +2 x1 (= own + 5) |
| real + another identity's audio | 1.06 | 13.93 | scattered -15..+15 |

Per-clip own LSE-C 4.2-8.8 (conversational clips with listening pauses score lower than LRS2-style speech); mismatched 0.5-1.8, so
every clip separates. The shift cases move each clip's offset by exactly 5 from its own value (23 of 24; M075, whose own offset
wavers between 0 and -1, gives -5 instead of -6). MediaPipe box vs S3FD changes LSE-C
by 0.02 on average.

Landmark sanity: `real` against its own lm.npz = 0.000 on every landmark metric (det 1.0, corr 1.0) — lmk.py is deterministic.
Noise floor `real_crf28` (H.264 crf 28 re-encode of real): face_nme 0.46, mouth_nme 0.50, pose_err 0.69, mouth_open_corr 0.988,
LSE-C 5.99.

## Reference rows (results/table.json)
| run | LSE-C | LSE-D | offset | face_nme | mouth_nme | pose_err | open_corr | det | realism | fooled |
|---|---|---|---|---|---|---|---|---|---|---|
| real (ceiling) | 6.16 | 7.83 | -0.50 | 0 | 0 | 0 | 1.000 | 1.0 | 8.08 | 0.458 (identical strips = chance) |
| real_crf28 (noise floor) | 5.99 | 7.92 | -0.50 | 0.46 | 0.50 | 0.69 | 0.988 | 1.0 | - | - |
| render2real_fast (render2real turbo_lora_ema) | 3.11 | 9.94 | -0.42 | 2.07 | 2.03 | 2.45 | 0.931 | 1.0 | 5.42 (real 8.13) | 0.000 |

render2real_fast was generated from the RENDER, so its landmark error vs the real landmarks measures the whole render pipeline.

## Runtime
Without judge, cold (no cache), 12 workers: 100 s for `real` and 373-398 s for `real_crf28` / `render2real_fast` while the host's load
average was 800-1250 (other agents' jobs); on a quiet box a clip is ~40 s lmk.py (heavy pose model, sequential VIDEO mode) + ~20 s
SyncNet, so an estimated ~1-1.5 min per run (not measured on an idle hub tonight). Fully cached re-run: 10 s. Judge: 48 calls, 5-8 min at 4 parallel; cached afterwards.

## Files
run_eval.py, sync.py, lm_metrics.py, judge.py, table.py, validate.py; third_party/{syncnet_python (git clone, code read before use),
weights/syncnet_v2.model (sha256 961e8696...), weights/sfd_face.pth (d54a87c2...), pylib/python_speech_features 0.6};
ref/real (symlinks to ROOT/data/clips/<id>/real.mp4), ref/real_crf28; judge_work/ (strips), scratch/ (dev audio, logs).
