#!/usr/bin/env python3
"""Merge eval/results/*.json -> eval/results/table.json (one row per run) for the page builder."""
import glob, json, os, time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
KEYS = ('lse_c', 'lse_d', 'offset', 'face_nme', 'mouth_nme', 'pose_err', 'mouth_open_corr', 'det_rate', 'realism', 'fooled_rate')


def row(doc):
    m = doc['mean']; s = m.get('syncnet') or {}; l = m.get('landmarks') or {}; j = m.get('realism') or {}
    r = dict(run=doc['run'], lse_c=s.get('lse_c'), lse_d=s.get('lse_d'), offset=s.get('offset_frames'),
             face_nme=l.get('face_nme'), mouth_nme=l.get('mouth_nme'), pose_err=l.get('pose_err'),
             mouth_open_corr=l.get('mouth_open_corr'), det_rate=l.get('det_rate'),
             realism=j.get('score_1_10'), fooled_rate=j.get('fooled_rate'))
    r['extra'] = dict(abs_offset=s.get('abs_offset_frames'), face_nme_pen=l.get('face_nme_pen'), mouth_nme_pen=l.get('mouth_nme_pen'),
                      pose_err_pen=l.get('pose_err_pen'), realism_real=j.get('score_real_1_10'), judge_trials=j.get('trials'),
                      n_clips=m.get('n_clips'), n_failed=m.get('n_failed'), made=doc.get('made'), dir=doc.get('dir'),
                      inputs=doc.get('inputs'), secs_metrics=doc.get('secs_metrics'))
    return r


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RES, '*.json'))):
        if os.path.basename(p) == 'table.json': continue
        try: rows.append(row(json.load(open(p))))
        except Exception as e: print('skip', p, e)
    doc = dict(made=time.strftime('%F %T'), columns=list(KEYS), rows=rows)
    out = os.path.join(RES, 'table.json'); tmp = out + '.tmp'
    json.dump(doc, open(tmp, 'w'), indent=1); os.replace(tmp, out)
    for r in rows: print(r['run'].ljust(24), ' '.join(f'{k}={r[k]}' for k in KEYS))
    print('wrote', out)


if __name__ == '__main__':
    main()
