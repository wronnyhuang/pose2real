#!/bin/bash
# Download landmark/skeleton clips, weights and eval outputs. usage: scripts/01_fetch_artifacts.sh <drive-folder-url-or-id>
set -e; cd "$(dirname "$0")/.."
mkdir -p artifacts/pose2real
if ! ls artifacts/pose2real/*.tar >/dev/null 2>&1; then
  # the Drive folder is access-restricted: sign in to Google in a browser and download the .tar files into artifacts/pose2real/ yourself,
  # or, if the folder owner enabled link sharing, this works unattended:
  pip install -q gdown==5.2.0 && gdown --folder "$1" -O artifacts
fi
for t in artifacts/pose2real/*.tar; do [ -f "$t" ] && tar -xf "$t"; done   # -> data/clips/<id>/{lm.npz,meta.json}, weights/<run>/last.pt
echo "real.mp4 / audio.wav are NOT redistributed: run scripts/02_data.sh --media-only to cut them from your own Seamless Interaction copy"
