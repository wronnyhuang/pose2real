#!/usr/bin/env python3
"""Write runs/<run>/run.json (labels for the page) from cfg.json/infer.json + the table below; make <run>__<mode> alias dirs."""
import json, os, glob
R = __import__('os').path.expandvars('$WORK/runs')
# run -> (group, label)
META = {
 'base': ('injection', 'Specified design: embedding added to the initial noise (splat encoder, 6 layers, 126 clips)'),
 'inj_early': ('injection', 'Added after the first conv layer ("early latent")'),
 'inj_controlnet': ('injection', 'ControlNet-style: multi-scale residuals into the UNet encoder'),
 'inj_xattn': ('injection', 'Cross-attention only: tokens, no spatial map'),
 'base_ownref': ('identity reference', 'Reference = first frame of the same clip (vs another window of the person)'),
 'enc_depth2': ('encoder', 'Encoder depth 2'),
 'enc_depth12': ('encoder', 'Encoder depth 12'),
 'enc_raster': ('encoder', 'No deep point encoder: landmarks drawn as fixed heat maps + small audio net'),
 'enc_vec': ('encoder', 'Pure vector encoder: no spatial splat, learned 12x12 queries upsampled'),
 'au_ctx': ('audio / lookahead', 'HuBERT layer-9 features over the whole clip (unlimited lookahead; 3,100 steps)'),
 'data_all': ('data', 'Same as base, trained on all 853 clips'),
 'bb_sd15': ('backbone', 'SD 1.5 (not distilled) used one-step'),
 'bb_tiny': ('backbone', 'Tiny-SD (distilled small UNet) one-step'),
 'v5_sync': ('lip-sync loss', 'v3 + 45 min, SyncNet loss on the whole face crop (cheats: eye sparkle)'),
 'v6_mouthsync': ('lip-sync loss', 'v3 + 45 min, SyncNet loss allowed to change the mouth box only'),
 'v3_seen': ('best', 'v3, seen identity: v2 + 50 more minutes, also trained on OTHER windows of the 8 test people'),
 'v4_more': ('best', 'v4, unseen identity: v2 + 50 more minutes on the same training set'),
 'v2': ('best', 'v2: all data + causal HuBERT + reference-anchored mouth + pre-trained encoder'),
}
for d in sorted(glob.glob(R + '/*/')):
    run = os.path.basename(d.rstrip('/'))
    if '__' in run or not os.path.exists(d + 'cfg.json'): continue
    cfg = json.load(open(d + 'cfg.json'))
    inf = json.load(open(d + 'infer.json')) if os.path.exists(d + 'infer.json') else {}
    g, lab = META.get(run, ('other', run))
    base = dict(group=g, label=lab, backbone=cfg.get('backbone'), injection=cfg.get('inj'), encoder=f"{cfg.get('enc')} d{cfg.get('depth')}",
                audio=cfg.get('audio'), audio_lookahead_frames=inf.get('test_au_la', cfg.get('au_la')), lm_lookahead_frames=inf.get('test_lm_la', cfg.get('lm_la')),
                ms_per_frame=inf.get('ms_per_frame'), gpu=inf.get('gpu'), train_minutes=round(cfg.get('train_minutes', 0), 1), steps=cfg.get('steps'),
                clips=cfg.get('clips'), mode='all landmarks + audio')
    if os.path.exists(d + 'bench.json'):
        b = json.load(open(d + 'bench.json'))
        if b.get('compiled_ms'): base['ms_per_frame'] = b['compiled_ms']; base['gpu'] = b['gpu'] + ' (LoRA fused + torch.compile)'
    json.dump(base, open(d + 'run.json', 'w'), indent=1)
    for od in sorted(glob.glob(d + 'out_*')):
        mode = os.path.basename(od)[4:]
        al = f'{R}/{run}__{mode}'
        os.makedirs(al, exist_ok=True)
        if not os.path.islink(al + '/out'): os.symlink(od, al + '/out')
        x = dict(base)
        tag_inf = d + 'infer_' + mode.split('_', 1)[1] + '.json' if '_' in mode else None
        if tag_inf and os.path.exists(tag_inf):
            ti = json.load(open(tag_inf)); x['audio_lookahead_frames'] = ti.get('test_au_la'); x['lm_lookahead_frames'] = ti.get('test_lm_la')
        if mode.startswith('alips'):
            x['mode'] = 'lips from audio (lower-face landmarks hidden)'; x['label'] = lab + ' — lips from audio'
            x['group'] = g if g in ('best', 'audio / lookahead', 'lip-sync loss') else 'lips from audio only (older runs)'
        if '_la' in mode: x['label'] += f" (audio look-ahead {mode.split('_la')[1]} frames = {int(mode.split('_la')[1]) * 33} ms)"
        if mode == 'ema': x['label'] = lab + ' + 35% previous-latent smoothing'; x['mode'] = 'all landmarks + audio, temporal smoothing'
        json.dump(x, open(al + '/run.json', 'w'), indent=1)
os.makedirs(R + '/render2real_fast', exist_ok=True)
json.dump({'ms_per_frame': 21.3, 'gpu': 'RTX 5090 (compiled)', 'backbone': 'sd-turbo', 'mode': 'render in, photo out'}, open(R + '/render2real_fast/run.json', 'w'))
print('ok')
