#!/bin/bash
# Download the pose2real artifact bundles from the public Google Drive folder BY FILE ID (no sign-in needed) and extract them at the repo root.
# usage: bash scripts/01_fetch_artifacts.sh [bundle.tar ...]     (default: every bundle in scripts/artifacts.tsv; see ARTIFACTS.md)
set -e; cd "$(dirname "$0")/.."
command -v gdown >/dev/null || pip install -q gdown==5.2.0
mkdir -p artifacts/pose2real
want=("$@")
while IFS=$'\t' read -r name id size; do
  [ -n "$name" ] || continue
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " == *" $name "* ]]; then continue; fi
  f=artifacts/pose2real/$name
  if [ "$(stat -c %s "$f" 2>/dev/null)" != "$size" ]; then
    echo "downloading $name ($((size/1000000)) MB)"; rm -f "$f"
    gdown "$id" -O "$f"
    [ "$(stat -c %s "$f")" = "$size" ] || { echo "size mismatch for $name (Drive quota page? retry later, or open the folder link in ARTIFACTS.md)"; exit 1; }
  fi
  tar -xf "$f" && echo "extracted $name"
done < scripts/artifacts.tsv
echo "fetch done. Real-person videos/audio are NOT redistributed: see DATA.md"
