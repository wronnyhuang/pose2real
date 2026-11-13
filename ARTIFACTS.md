# Downloadable artifacts (pose2real)

Trained weights, generated training data and eval outputs are too large for git. They live in a Google Drive folder
**"Avatar photoreal research artifacts"/pose2real/** (https://drive.google.com/drive/folders/1G7jieF9XQvVJ9VosnpVx-EMzMBytcfAg), access-restricted (shared per account).
`MANIFEST.md` in that folder lists every bundle with size, file counts and contents; a copy is in this repo (`MANIFEST.md`).

Download the `.tar` files for this repo into `artifacts/pose2real/` and run `bash scripts/01_fetch_artifacts.sh` (it extracts them at the repo root).
What is deliberately not in the bundles: real-person videos and voices (re-created from Meta Seamless Interaction by `scripts/02_data.sh`, see `DATA.md`) and base-model weights (Hugging Face ids in `README.md`).
