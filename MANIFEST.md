# MANIFEST - Avatar photoreal research artifacts

Folder layout: `render2real/` and `pose2real/`, each a set of `.tar` bundles (extract at the repo root of the matching GitHub repo: `scripts/01_fetch_artifacts.sh` does it). Sizes in bytes / MB.


## render2real/

### r2r_eval_outputs.tar - 179.2 MB, 290 files
Generated eval videos (fast/out/<variant>, wan/out/<variant>: heldout clips, avatar render clips, a few train samples), latency/flicker json, and per-clip metric + judge results (eval_results/*.json).

- `./eval_results/*.json` - 31 files, 1.0 MB
- `./fast/*.json` - 2 files, 0.0 MB
- `./fast/out/*.mp4` - 160 files, 122.2 MB
- `./wan/out/*.mp4` - 97 files, 55.7 MB
- `./wan/out/lora_prompt_cv6/M100_P1067.mp4` 1019554 B
- `./wan/out/lora_prompt_cv6/M099_P1067.mp4` 1130317 B
- `./wan/out/lora_raw_s20/M009_P0217.mp4` 1455201 B
- `./wan/out/lora_raw_s20/M001_P0115.mp4` 1247862 B
- `./wan/out/lora_raw_s20/M006_P0216_trainsample.mp4` 1134983 B
- `./fast/out/turbo_lora/muse_04.mp4` 1025725 B
- `./fast/out/turbo_lora/M100_P1067.mp4` 1020984 B
- `./fast/out/turbo_lora/muse_01.mp4` 1176636 B
- `./fast/out/turbo_lora/M001_P0115.mp4` 1281920 B
- `./fast/out/turbo_lora/M075_P0845.mp4` 1226781 B
- `./fast/out/turbo_lora/M099_P1067.mp4` 1144352 B
- `./fast/out/turbo_lora/muse_02.mp4` 1258150 B
- `./fast/out/turbo_lora/M092_P0989.mp4` 1083515 B
- `./fast/out/turbo_lora/muse_03.mp4` 1159542 B
- `./fast/out/base_s075/juno_03.mp4` 1004940 B
- `./fast/out/base_s075/M023_P0370.mp4` 1024715 B
- `./fast/out/base_s075/M075_P0845.mp4` 1068362 B
- `./fast/out/base_s075/juno_02.mp4` 1162502 B
- `./fast/out/base_s075/M092_P0989.mp4` 1271754 B
- `./fast/out/base_s075/M002_P0116_trainsample.mp4` 1064508 B
- `./fast/out/base_s075/M037_P0485.mp4` 1002242 B
- `./fast/out/base_asft/M100_P1067.mp4` 1045867 B
- `./fast/out/base_asft/M001_P0115.mp4` 1205864 B
- `./fast/out/base_asft/M075_P0845.mp4` 1066635 B
- `./fast/out/base_asft/M099_P1067.mp4` 1122548 B
- `./fast/out/turbo_lora_ema/muse_01.mp4` 1040646 B
- `./fast/out/turbo_lora_ema/M001_P0115.mp4` 1110203 B
- `./fast/out/turbo_lora_ema/M075_P0845.mp4` 1099639 B
- `./fast/out/turbo_lora_ema/M099_P1067.mp4` 1016894 B
- `./fast/out/turbo_lora_ema/muse_02.mp4` 1114909 B
- `./fast/out/turbo_lora_ema/muse_03.mp4` 1026297 B

### r2r_training_pairs_render_side.tar - 44.8 MB, 313 files
Training pairs, render side only: data/pairs/<id>/render.mp4 (stylised 3D render, 384x384, 30 fps, 8 s) + meta.json (source file id, start time, crop box) + data/manifest.json (split, identity, QC). The matching real.mp4 targets are NOT redistributed (Seamless Interaction footage) - re-cut with scripts/02_data.sh --targets-only.

- `data/*.json` - 1 files, 0.0 MB
- `data/pairs/*.json` - 156 files, 0.1 MB
- `data/pairs/*.mp4` - 156 files, 44.4 MB

### r2r_weights_fast.tar - 185.6 MB, 3 files
Trained weights of the fast model family: turbo_lora.pt (main one-step SD-Turbo LoRA + skip-VAE; load with Turbo().load_trainable), turbo_lora_temporal.pt (failed temporal variant), unet_l1lpips.pt (370 fps U-Net). Base model stabilityai/sd-turbo is NOT included (HF).

- `fast/weights/turbo_lora.pt` 38165257 B
- `fast/weights/unet_l1lpips.pt` 109259839 B
- `fast/weights/turbo_lora_temporal.pt` 38211337 B

### r2r_weights_wan_lora.tar - 43.8 MB, 2 files
Wan 2.1 VACE 1.3B LoRA (rank 32, epoch 1) + training_args.json. Base Wan weights not included (HF ids in README).

- `wan/lora/training_args.json` 2081 B
- `wan/lora/vace_lora_r32_epoch1.safetensors` 43781016 B


## pose2real/

### p2r_data_landmarks.tar - 1267.7 MB, 1775 files
Conditioning features per clip: data/clips/<id>/lm.npz (MediaPipe per frame: face 478x3, pose 33x4, 52 blendshapes, flags) + meta.json (source id, start time, crop, split) for 887 clips; data/manifest.json. Videos and audio.wav (real people) are NOT included; audio features are recomputed from audio with HuBERT (data.py). Rebuild media with scripts/02_data.sh --media-only.

- `data/*.json` - 1 files, 0.1 MB
- `data/clips/*.json` - 887 files, 0.8 MB
- `data/clips/*.npz` - 887 files, 1265.1 MB
- `data/clips/X0333_P0801/lm.npz` 1428507 B
- `data/clips/X0462_P1276A/lm.npz` 1427643 B
- `data/clips/X0456_P0780/lm.npz` 1421485 B
- `data/clips/X0611_P1276A/lm.npz` 1426596 B
- `data/clips/X0182_P1275A/lm.npz` 1426955 B
- `data/clips/X0131_P1347A/lm.npz` 1426347 B
- `data/clips/X0715_P1277A/lm.npz` 1428253 B
- `data/clips/M100_P1067/lm.npz` 1427690 B
- `data/clips/X0065_P1347A/lm.npz` 1427084 B
- `data/clips/X0394_P0852/lm.npz` 1424644 B
- `data/clips/M132_P1346A/lm.npz` 1428582 B
- `data/clips/X0207_P1276A/lm.npz` 1427592 B
- `data/clips/X0503_P0410/lm.npz` 1427562 B
- `data/clips/X0590_P0383/lm.npz` 1426272 B
- `data/clips/X0627_P1281A/lm.npz` 1424559 B
- `data/clips/X0141_P0663/lm.npz` 1428165 B
- `data/clips/M026_P0383/lm.npz` 1426193 B
- `data/clips/X0292_P0852/lm.npz` 1424939 B
- `data/clips/X0673_P1282A/lm.npz` 1424720 B
- `data/clips/X0188_P1347A/lm.npz` 1427726 B
- `data/clips/X0633_P0410/lm.npz` 1425888 B
- `data/clips/X0223_P0799/lm.npz` 1428498 B
- `data/clips/M112_P1276A/lm.npz` 1425805 B
- `data/clips/X0074_P0566/lm.npz` 1427435 B
- `data/clips/X0658_P0844A/lm.npz` 1428998 B
- `data/clips/X0265_P0731/lm.npz` 1427494 B
- `data/clips/M126_P1282A/lm.npz` 1427099 B
- `data/clips/X0193_P0410/lm.npz` 1425243 B
- `data/clips/X0536_P0844A/lm.npz` 1426767 B
- `data/clips/X0552_P0852/lm.npz` 1425035 B
- `data/clips/X0619_P0410/lm.npz` 1425960 B
- `data/clips/X0427_P0852/lm.npz` 1423882 B
- `data/clips/X0097_P1346A/lm.npz` 1426409 B
- `data/clips/X0494_P1276A/lm.npz` 1426193 B
- `data/clips/X0667_P0410/lm.npz` 1427461 B
- `data/clips/X0391_P0780/lm.npz` 1426376 B
- `data/clips/X0682_P1277A/lm.npz` 1425062 B
- `data/clips/X0646_P0801/lm.npz` 1428208 B
- `data/clips/X0399_P1281A/lm.npz` 1426324 B
- `data/clips/X0542_P1281A/lm.npz` 1426854 B
- `data/clips/X0727_P1347A/lm.npz` 1426320 B
- `data/clips/X0540_P1276A/lm.npz` 1426705 B
- `data/clips/X0094_P1277A/lm.npz` 1425953 B
- `data/clips/X0104_P0383/lm.npz` 1426868 B
- `data/clips/X0687_P0350/lm.npz` 1428720 B
- `data/clips/M072_P0844A/lm.npz` 1428108 B
- `data/clips/X0635_P0844A/lm.npz` 1428460 B
- `data/clips/X0016_P0840/lm.npz` 1426444 B
- `data/clips/M089_P0947/lm.npz` 1427825 B
- `data/clips/X0230_P1276A/lm.npz` 1426212 B
- `data/clips/X0518_P0383/lm.npz` 1426581 B
- `data/clips/X0179_P0947/lm.npz` 1426153 B
- `data/clips/X0668_P0801/lm.npz` 1427519 B
- `data/clips/R015_P0989/lm.npz` 1425487 B
- `data/clips/M115_P1277A/lm.npz` 1427515 B
- `data/clips/X0569_P1276A/lm.npz` 1427430 B
- `data/clips/X0354_P0801/lm.npz` 1427528 B
- `data/clips/X0363_P1282A/lm.npz` 1426899 B
- `data/clips/X0098_P1347A/lm.npz` 1427368 B
- `data/clips/X0057_P1066/lm.npz` 1428894 B

### p2r_eval_outputs.tar - 308.9 MB, 548 files
Generated videos for every reported run (runs/<run>/out = true landmarks, out_alips = lips from audio only), their cfg/infer/bench json, and eval_results/*.json incl. table.json.

- `./eval_results/*.json` - 43 files, 0.7 MB
- `./runs/*.json` - 37 files, 0.0 MB
- `./runs/au_ctx/*.mp4` - 24 files, 17.2 MB
- `./runs/base/*.mp4` - 24 files, 15.5 MB
- `./runs/base_ownref/*.mp4` - 24 files, 15.5 MB
- `./runs/bb_sd15/*.mp4` - 24 files, 16.0 MB
- `./runs/bb_tiny/*.mp4` - 24 files, 12.6 MB
- `./runs/data_all/*.mp4` - 24 files, 16.2 MB
- `./runs/enc_depth12/*.mp4` - 24 files, 15.8 MB
- `./runs/enc_depth2/*.mp4` - 24 files, 16.4 MB
- `./runs/enc_raster/*.mp4` - 24 files, 16.5 MB
- `./runs/inj_controlnet/*.mp4` - 24 files, 12.7 MB
- `./runs/inj_early/*.mp4` - 24 files, 8.2 MB
- `./runs/inj_xattn/*.mp4` - 24 files, 13.3 MB
- `./runs/v2/*.mp4` - 72 files, 45.1 MB
- `./runs/v3_seen/*.mp4` - 36 files, 25.6 MB
- `./runs/v4_more/*.mp4` - 24 files, 19.7 MB
- `./runs/v5_sync/*.mp4` - 24 files, 21.1 MB
- `./runs/v6_mouthsync/*.mp4` - 24 files, 20.4 MB
- `./runs/v5_sync/out_alips/M001_P0115.mp4` 1099465 B
- `./runs/v5_sync/out_alips/M075_P0845.mp4` 1091010 B
- `./runs/v5_sync/out_alips/M099_P1067.mp4` 1012990 B
- `./runs/v5_sync/out_alips/M092_P0989.mp4` 1164251 B
- `./runs/v4_more/out_alips/M075_P0845.mp4` 1049889 B
- `./runs/v4_more/out_alips/M099_P1067.mp4` 1118237 B
- `./runs/v4_more/out/M075_P0845.mp4` 1053992 B
- `./runs/v4_more/out/M099_P1067.mp4` 1121855 B

### p2r_weights.tar - 860.8 MB, 9 files
pose2real weights: base (126 clips), v2 (853 clips), v3_seen, v5_sync (SyncNet-loss, reward-hacks), each with cfg.json + last.pt (LoRA + conditioning encoder + GAN-free inference weights), and xy_long/enc.pt (pre-trained landmark/audio encoder).

- `weights/v3_seen/cfg.json` 607 B
- `weights/v3_seen/last.pt` 200465987 B
- `weights/v5_sync/cfg.json` 615 B
- `weights/v5_sync/last.pt` 200465987 B
- `weights/base/cfg.json` 492 B
- `weights/base/last.pt` 200055624 B
- `weights/v2/cfg.json` 572 B
- `weights/v2/last.pt` 200465987 B
- `weights/xy_long/enc.pt` 59364317 B


Total: 2.89 GB (excluding this file).


## Not included
- Real videos / voices of Seamless Interaction participants (real.mp4, audio.wav, torso-up clips): re-created from the dataset by the scripts; see DATA.md in each repo (CC BY-NC 4.0).
- Base model weights (SD-Turbo, Wan 2.1, HuBERT, MediaPipe task files, SyncNet): download scripts/HF ids in the READMEs.
- torso_up (512x640 + hands) weights and landmark clips: that experiment was still running at export time.
- Avatar-session screen recordings and the avatar render clips used only as demo inputs.
