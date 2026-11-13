#!/usr/bin/env python3
"""Merge eval/results/*.json -> eval/results/table.json (one row per run) for the page builder.
Torso rows (framing "torso", 20 heldout clips) + bust_* rows copied from pose2real/eval/results/table.json (framing "bust",
12 heldout clips, no arm/hand fields)."""
import glob, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
PRIOR_TABLE = __import__('os').path.expandvars('$WORK/eval/results/table.json')
BUST_RUNS = ('real', 'real_crf28', 'render2real_fast', 'base', 'inj_controlnet', 'enc_raster', 'v2', 'v3_seen', 'v4_more')
KEYS = ('lse_c', 'lse_d', 'offset', 'face_nme', 'mouth_nme', 'mouth_open_corr', 'det_rate', 'pose_err',
        'shoulder_err', 'elbow_err', 'wrist_err', 'arm_err',
        'hand_det_rate', 'hand_extra_rate', 'hand_err', 'hand_err_hs', 'hand_err_pen', 'finger_pose_err', 'finger_angle_err_deg',
        'realism', 'realism_real', 'fooled_rate', 'hands_score', 'hands_score_real')


def flat(m):
    s = m.get('syncnet') or {}; l = m.get('landmarks') or {}; h = m.get('hands') or {}; j = m.get('realism') or {}
    r = dict(lse_c=s.get('lse_c'), lse_d=s.get('lse_d'), offset=s.get('offset_frames'), abs_offset=s.get('abs_offset_frames'))
    for k in ('face_nme', 'mouth_nme', 'mouth_open_corr', 'det_rate', 'pose_err', 'shoulder_err', 'elbow_err', 'wrist_err',
              'arm_err', 'face_nme_pen', 'mouth_nme_pen', 'pose_err_pen'):
        r[k] = l.get(k)
    for k in ('hand_det_rate', 'hand_extra_rate', 'hand_err', 'hand_err_hs', 'hand_err_pen', 'hand_err_hs_pen', 'finger_pose_err',
              'finger_angle_err_deg', 'n_hand_frames', 'n_hand_both', 'n_hand_clips'):
        r[k] = h.get(k)
    r.update(realism=j.get('score_1_10'), realism_real=j.get('score_real_1_10'), fooled_rate=j.get('fooled_rate'),
             judge_trials=j.get('trials'), hands_score=j.get('hands_score_1_10'), hands_score_real=j.get('hands_score_real_1_10'),
             hands_judge_clips=j.get('hands_clips'), n_clips=m.get('n_clips'), n_failed=m.get('n_failed'))
    return r


def row(doc):
    r = dict(run=doc['run'], framing='torso', **flat(doc['mean']))
    for s in ('same12', 'gesture8'):
        if 'mean_' + s in doc: r[s] = flat(doc['mean_' + s])
    r['extra'] = dict(made=doc.get('made'), dir=doc.get('dir'), inputs=doc.get('inputs'), secs_metrics=doc.get('secs_metrics'),
                      secs_total=doc.get('secs_total'), sync_interp=doc.get('sync_interp'), failed=doc.get('failed'))
    return r


def bust_rows():
    try: rows = json.load(open(PRIOR_TABLE))['rows']
    except Exception as e: print('no prior table', e); return []
    out = []
    for r in rows:
        if r['run'] not in BUST_RUNS: continue
        e = r.get('extra') or {}
        b = {k: None for k in KEYS}
        b.update(run='bust_' + r['run'], framing='bust', lse_c=r.get('lse_c'), lse_d=r.get('lse_d'), offset=r.get('offset'),
                 abs_offset=e.get('abs_offset'), face_nme=r.get('face_nme'), mouth_nme=r.get('mouth_nme'),
                 mouth_open_corr=r.get('mouth_open_corr'), det_rate=r.get('det_rate'), pose_err=r.get('pose_err'),
                 face_nme_pen=e.get('face_nme_pen'), mouth_nme_pen=e.get('mouth_nme_pen'), pose_err_pen=e.get('pose_err_pen'),
                 realism=r.get('realism'), realism_real=e.get('realism_real'), fooled_rate=r.get('fooled_rate'),
                 judge_trials=e.get('judge_trials'), n_clips=e.get('n_clips'), n_failed=e.get('n_failed'))
        b['extra'] = dict(source=PRIOR_TABLE, made=e.get('made'), dir=e.get('dir'),
                          note='pose2real bust framing (384x384, 12 heldout clips); pose_err there = points 0-16 incl. mostly '
                               'invisible elbows/wrists; realism judged on 6-frame 384px strips')
        out.append(b)
    out.sort(key=lambda r: BUST_RUNS.index(r['run'][5:]))
    return out


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, '*.json'))):
        if os.path.basename(p) == 'table.json': continue
        try: rows.append(row(json.load(open(p))))
        except Exception as e: print('skip', p, e)
    rows += bust_rows()
    doc = dict(made=time.strftime('%F %T'), columns=list(KEYS),
               notes='torso rows: mean over the 20 heldout clips (row.same12 / row.gesture8 = subsets); hand metrics are '
                     'hand-instance-weighted (README). bust_* rows: pose2real, 12 clips, framing bust.', rows=rows)
    out = os.path.join(RES, 'table.json'); tmp = out + f'.{os.getpid()}.tmp'
    json.dump(doc, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    for r in rows: print(r['run'].ljust(26), ' '.join(f'{k}={r.get(k)}' for k in KEYS if r.get(k) is not None))
    print('wrote', out)


if __name__ == '__main__':
    main()
