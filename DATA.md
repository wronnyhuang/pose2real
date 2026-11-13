# Data sources and licences

* **Meta Seamless Interaction** (https://github.com/facebookresearch/seamless_interaction, dataset `facebook/seamless-interaction` on Hugging Face):
  the only source of real footage and speech. Released by Meta under **CC BY-NC 4.0** (attribution, non-commercial) — check the dataset card
  for the current terms. Only the `naturalistic` / `improvised` dev splits were used. **This repository contains no Seamless videos or audio**;
  clip ids (`P####` participant ids are the dataset's own anonymised ids), crop boxes, start times and QC numbers are in `meta.json` / `manifest.json`,
  and the scripts re-cut the clips from your own licensed copy. Anything derived from it (weights, generated videos) inherits the non-commercial restriction.
* **Models**: Stable Diffusion Turbo (`stabilityai/sd-turbo`, Stability AI community / non-commercial research terms — see model card),
  Wan 2.1 (Apache-2.0), CausVid LoRA via `Kijai/WanVideo_comfy`, HuBERT base (Apache-2.0), MediaPipe landmarker task files (Apache-2.0),
  SyncNet (joonson/syncnet_python), DiffSynth-Studio (Apache-2.0), insightface buffalo_l (non-commercial research). Check each card before commercial use.
* torso_up uses the same Seamless clips, cut as torso-up crops.
