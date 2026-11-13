#!/usr/bin/env python3
import json, os, glob
R = __import__('os').path.expandvars('$WORK/runs')
META = {
 't_base': ('ablation (45 min each)', 'Base: his design at 512x640 — face + skeleton + hand landmarks and audio through a 6-layer encoder, added to the starting noise; hand-crop loss on'),
 't_nohands': ('ablation (45 min each)', 'No hand landmarks in the encoder (skeleton wrist points only)'),
 't_whand0': ('ablation (45 min each)', 'No hand-crop loss (hand landmarks still fed in); RTX 3090, full data'),
 't_s384': ('ablation (45 min each)', 'Lower resolution 384x512 (upscaled to 512x640 for scoring); RTX 3090, full data'),
 't_inj_controlnet': ('ablation (45 min each)', 'ControlNet-style injection instead of adding to the noise'),
 't_enc_raster': ('ablation (45 min each)', 'No deep point encoder: landmarks and hands drawn as fixed heat maps'),
 't_depth2': ('ablation (45 min each)', 'Encoder depth 2 instead of 6; RTX 3090, full data'),
 't_hand_gan': ('ablation (45 min each)', 'Extra discriminator on hand crops'),
 't_v2': ('best', 'v2 (unseen people): all data, causal HuBERT audio, reference-anchored mouth, 60 min'),
 't_v3_seen': ('best', 'v3 (seen people): v2 + 45 min, also trained on OTHER footage of the 8 test people'),
 't_v4_long': ('best', 'v4 (seen people): v3 + 90 more minutes'),
}
for d in sorted(glob.glob(R + '/*/')):
    run = os.path.basename(d.rstrip('/'))
    if run not in META or not os.path.exists(d + 'cfg.json'): continue
    cfg = json.load(open(d + 'cfg.json')); inf = json.load(open(d + 'infer.json')) if os.path.exists(d + 'infer.json') else {}
    b = json.load(open(d + 'bench.json')) if os.path.exists(d + 'bench.json') else {}
    g, lab = META[run]
    gpu = (b.get('gpu') or inf.get('gpu') or '').replace('NVIDIA GeForce ', '')
    json.dump(dict(group=g, label=lab, ms_per_frame=b.get('compiled_ms') or b.get('plain_ms') or inf.get('ms_per_frame'), plain_ms=b.get('plain_ms'),
                   gpu=gpu + (' (compiled)' if b.get('compiled_ms') else ''), train_minutes=round(cfg.get('train_minutes', 0), 1), steps=cfg.get('steps'),
                   clips=cfg.get('clips'), size=cfg.get('size')), open(d + 'run.json', 'w'), indent=1)
    print(run, json.load(open(d + 'run.json'))['ms_per_frame'], gpu, cfg.get('steps'), cfg.get('clips'))
