# Downloadable artifacts (pose2real)

Weights, generated training data and eval outputs are too large for git. They are in a **public** Google Drive folder (anyone with the link can view and download):

* All bundles for both projects: https://drive.google.com/drive/folders/1G7jieF9XQvVJ9VosnpVx-EMzMBytcfAg  ("Avatar photoreal research artifacts")
* This repo's bundles: https://drive.google.com/drive/folders/1BRnk-NPvY58_4bmzLddzrf1G_XU1Tfkm  (`pose2real/`)
* `MANIFEST.md` (contents, file counts, sizes): https://drive.google.com/file/d/1fwbwSP7IZDvqsBCq6qT2Pi08hjz6yWXp/view  — a copy is in this repo as `MANIFEST.md`

| bundle | size | what is inside | direct link | file id |
|---|---|---|---|---|
| `p2r_weights.tar` | 821.0 MB (860,835,840 B) | trained run folders (`weights/<run>/{cfg.json,last.pt}`) | [open](https://drive.google.com/file/d/1lYEZ7arBH5ofa69pWL2UCi5ky8R0e0fs/view) · [download](https://drive.google.com/uc?export=download&id=1lYEZ7arBH5ofa69pWL2UCi5ky8R0e0fs) | `1lYEZ7arBH5ofa69pWL2UCi5ky8R0e0fs` |
| `p2r_data_landmarks.tar` | 1209.0 MB (1,267,722,240 B) | per-clip MediaPipe landmarks `lm.npz` + `meta.json` + manifest (no video/audio) | [open](https://drive.google.com/file/d/1tCWDuremVKbJqCxPh_UNw31jbLGY-7mH/view) · [download](https://drive.google.com/uc?export=download&id=1tCWDuremVKbJqCxPh_UNw31jbLGY-7mH) | `1tCWDuremVKbJqCxPh_UNw31jbLGY-7mH` |
| `p2r_eval_outputs.tar` | 294.6 MB (308,910,080 B) | generated eval videos + per-clip metric/judge JSON | [open](https://drive.google.com/file/d/1XeXgjBlhWgFdK4xFPkxcuvKm_q5epAnF/view) · [download](https://drive.google.com/uc?export=download&id=1XeXgjBlhWgFdK4xFPkxcuvKm_q5epAnF) | `1XeXgjBlhWgFdK4xFPkxcuvKm_q5epAnF` |

Fetch them all (downloads by file id with `gdown`, no Google sign-in; resumes by skipping complete files, extracts at the repo root):
```bash
bash scripts/01_fetch_artifacts.sh                 # everything
bash scripts/01_fetch_artifacts.sh p2r_weights.tar   # one bundle
```
The id/size list the script uses is `scripts/artifacts.tsv`. If Drive answers with a "too many downloads" page, retry later or download the files from the links above and put them in `artifacts/pose2real/`.

Deliberately not in the bundles: real-person videos and voices (re-created from Meta Seamless Interaction by `scripts/02_data.sh`, see `DATA.md`) and base-model weights (Hugging Face ids in `README.md`).
