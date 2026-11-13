#!/usr/bin/env python3
"""pose2real pose2real results page.

  python3 build_page.py                 # real build -> $OUT_DIR/pose2real/{index.html,videos/}
  python3 build_page.py --dummy         # build from page/dummy/ (made on demand) -> page/dummy_out/
  python3 build_page.py --no-videos     # html only (reuse existing videos)

Reads (all optional; the page says "no runs yet" where data is missing):
  ROOT/eval/results/table.json   rows: run, lse_c, lse_d, offset, face_nme, mouth_nme, pose_err, mouth_open_corr,
                                 det_rate, realism, fooled_rate (list, dict-of-rows or {"rows": [...]}; nested dicts ok)
  ROOT/runs/<run>/run.json       group, label, backbone, injection, encoder, audio_lookahead_frames, lm_lookahead_frames,
                                 ms_per_frame, train_minutes, steps
  ROOT/runs/<run>/out/<clip>.mp4, ROOT/data/clips/<clip>/{real.mp4,audio.wav,lm.npz}, ROOT/data/manifest.json
  ROOT/pods/spend.json
  PAGE/{summary,worked,didnt,next}.md, PAGE/showcase.json   (dummy mode: the same names inside page/dummy/)
NME / pose error values below 1 are shown x100 as percent (the eval harness stores fractions of the frame/face);
values already >= 1 are shown as is. Constant FRACTION_METRICS controls which columns.
"""
import argparse, html, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = os.path.dirname(HERE)
ARTIFACT = __import__('os').path.expandvars('$OUT_DIR/pose2real')
VENV_PY = __import__('os').path.expandvars('$WORK/eval/venv/bin/python')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
T4448_PAIRS = __import__('os').path.expandvars('$WORK/data/pairs')
SIZE, FPS, CRF = 384, 30, 23
RENDER_VERSION = 'v1'  # bump to force triptych re-encode
REFS = ('real', 'render2real_fast')
REF_LABELS = {'real': 'Real video (ceiling)', 'render2real_fast': 'Render → photoreal, fast model'}
FRACTION_METRICS = ('face_nme', 'mouth_nme', 'pose_err')

try:
    import numpy as np, cv2
except ImportError:
    if os.path.exists(VENV_PY) and os.environ.get('P2R_REEXEC') != '1':
        os.environ['P2R_REEXEC'] = '1'
        os.execv(VENV_PY, [VENV_PY] + sys.argv)
    raise


# ------------------------------------------------------------------ helpers
def jload(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return default


def tload(p):
    try:
        with open(p) as f: return f.read()
    except Exception: return None


def esc(s): return html.escape(str(s))


def inline(s):
    s = esc(s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*([^*\n]+)\*(?!\w)', r'<em>\1</em>', s)
    return s


def md(text):
    """markdown-lite: paragraphs, '- ' bullets, '1.' lists, #-headings, **bold**, *em*, `code`."""
    if not text or not text.strip(): return None
    out, lst, para = [], None, []
    def flush():
        nonlocal para
        if para: out.append('<p>' + inline(' '.join(para)) + '</p>'); para = []
    def close():
        nonlocal lst
        if lst: out.append(f'</{lst}>'); lst = None
    for ln in text.splitlines():
        s = ln.rstrip()
        m = re.match(r'^(#{1,4})\s+(.*)', s)
        b = re.match(r'^\s*[-*]\s+(.*)', s); n = re.match(r'^\s*\d+[.)]\s+(.*)', s)
        if not s.strip(): flush(); close(); continue
        if m: flush(); close(); lv = min(len(m.group(1)) + 2, 5); out.append(f'<h{lv}>{inline(m.group(2))}</h{lv}>')
        elif b or n:
            flush(); kind = 'ul' if b else 'ol'
            if lst != kind: close(); out.append(f'<{kind}>'); lst = kind
            out.append('<li>' + inline((b or n).group(1)) + '</li>')
        else: close(); para.append(s.strip())
    flush(); close()
    return '\n'.join(out)


def num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)) and v == v: return float(v)
    if isinstance(v, str):
        try: return float(v)
        except ValueError: return None
    return None


def find(d, *names):
    """first numeric value for any of names, looking in d then in its nested dicts (one level, or two)."""
    if not isinstance(d, dict): return None
    for n in names:
        if n in d and num(d[n]) is not None: return num(d[n])
    for v in d.values():
        if isinstance(v, dict):
            r = find(v, *names) if any(isinstance(x, dict) for x in v.values()) else \
                next((num(v[n]) for n in names if n in v and num(v[n]) is not None), None)
            if r is not None: return r
    return None


# ------------------------------------------------------------------ data loading
METRICS = [  # key, aliases, header, headline label, higher_is_better, decimals
    ('lse_c', ('lse_c', 'LSE-C', 'lse-c'), 'LSE-C ↑', 'Lip-sync confidence, SyncNet LSE-C (higher is better)', True, 2),
    ('lse_d', ('lse_d', 'LSE-D', 'lse-d'), 'LSE-D ↓', 'Lip-sync distance, SyncNet LSE-D (lower is better)', False, 2),
    ('offset', ('offset', 'offset_frames'), 'Sync offset (fr)', None, None, 1),
    ('face_nme', ('face_nme',), 'Face err % ↓', 'Face landmark error % (lower is better)', False, 2),
    ('mouth_nme', ('mouth_nme',), 'Mouth err % ↓', 'Mouth landmark error % (lower is better)', False, 2),
    ('pose_err', ('pose_err',), 'Skeleton err % ↓', 'Skeleton error % (lower is better)', False, 2),
    ('mouth_open_corr', ('mouth_open_corr',), 'Mouth-open corr ↑', None, True, 2),
    ('det_rate', ('det_rate',), 'Face found ↑', None, True, 2),
    ('realism', ('realism', 'score_1_10', 'realism_score'), 'Realism /10 ↑', 'Realism, judge score 1-10 (higher is better)', True, 1),
    ('fooled_rate', ('fooled_rate',), 'Fooled ↑', None, True, 2),
    ('ms_per_frame', ('ms_per_frame', 'ms_frame'), 'ms/frame ↓', 'Speed, ms per frame (lower is better)', False, 0),
]
MK = {m[0]: m for m in METRICS}
CONFIG_COLS = [('backbone', 'Backbone'), ('injection', 'Injection'), ('encoder', 'Encoder'),
               ('audio_lookahead_frames', 'Audio look-ahead'), ('lm_lookahead_frames', 'Landmark look-ahead'),
               ('train_minutes', 'Train min'), ('steps', 'Steps')]


def load_rows(root):
    raw = jload(os.path.join(root, 'eval', 'results', 'table.json'))
    items = []
    if isinstance(raw, dict) and isinstance(raw.get('rows'), (list, dict)): raw = raw['rows']
    if isinstance(raw, list):
        for r in raw:
            if isinstance(r, dict) and (r.get('run') or r.get('name')): items.append((str(r.get('run') or r.get('name')), r))
    elif isinstance(raw, dict):
        for k, r in raw.items():
            if isinstance(r, dict): items.append((str(r.get('run') or k), r))
    rows = {}
    for name, r in items:
        row = {'run': name}
        for key, al, *_ in METRICS: row[key] = find(r, *al)
        rows[name] = row
    # runs known only from runs/<run>/run.json
    rd = os.path.join(root, 'runs')
    if os.path.isdir(rd):
        for n in sorted(os.listdir(rd)):
            if os.path.exists(os.path.join(rd, n, 'run.json')) and n not in rows:
                rows[n] = {'run': n, **{k[0]: None for k in METRICS}}
    for name, row in rows.items():
        cfg = jload(os.path.join(rd, name, 'run.json'), {}) or {}
        row['cfg'] = cfg if isinstance(cfg, dict) else {}
        if row.get('ms_per_frame') is None: row['ms_per_frame'] = find(row['cfg'], 'ms_per_frame')
        row['group'] = 'reference rows' if name in REFS else str(row['cfg'].get('group') or 'other')
        row['label'] = REF_LABELS.get(name) or str(row['cfg'].get('label') or name)
        row['has_out'] = os.path.isdir(os.path.join(rd, name, 'out'))
    return rows


def disp(key, v):
    if v is None: return '<span class="mut">—</span>'
    nd = MK[key][5]
    if key in FRACTION_METRICS: return f'{v:.{nd}f}'
    return f'{v:.{nd}f}'


def normalise_percent(rows):
    """NME / pose error stored as fractions (<1 everywhere) -> percent."""
    for key in FRACTION_METRICS:
        vals = [r[key] for r in rows.values() if r.get(key) is not None]
        if vals and max(vals) < 1.0:
            for r in rows.values():
                if r.get(key) is not None: r[key] *= 100


def score(rows, key):
    """best-first order key for a metric"""
    hib = MK[key][4]
    return lambda v: -v if hib else v


def best_run(rows):
    cand = [r for n, r in rows.items() if n not in REFS and r.get('lse_c') is not None]
    return max(cand, key=lambda r: r['lse_c']) if cand else None


# ------------------------------------------------------------------ conditioning panel
CONN = jload(os.path.join(HERE, 'connections.json'))
# BGR
C_CONTOUR, C_EYE, C_LIPS, C_IRIS, C_POSE = (255, 230, 90), (255, 255, 255), (60, 230, 255), (255, 255, 255), (0, 140, 255)


def draw_cond(face, pose, face_ok, pose_ok, size=SIZE):
    img = np.zeros((size, size, 3), np.uint8)
    def seg(pts, edges, col, th):
        for a, b in edges:
            p, q = pts[a], pts[b]
            cv2.line(img, (int(round(p[0] * size)), int(round(p[1] * size))), (int(round(q[0] * size)), int(round(q[1] * size))), col, th, cv2.LINE_AA)
    if pose_ok:
        vis = pose[:, 3] > 0.5
        edges = [e for e in CONN['POSE'] if vis[e[0]] and vis[e[1]]]
        seg(pose, edges, C_POSE, 3)
        for i in np.nonzero(vis)[0]:
            cv2.circle(img, (int(round(pose[i, 0] * size)), int(round(pose[i, 1] * size))), 4, C_POSE, -1, cv2.LINE_AA)
    if face_ok:
        seg(face, CONN['CONTOURS'], C_CONTOUR, 1)
        for k in ('LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYEBROW', 'RIGHT_EYEBROW'): seg(face, CONN[k], C_EYE, 1)
        for k in ('LEFT_IRIS', 'RIGHT_IRIS'): seg(face, CONN[k], C_IRIS, 1)
        seg(face, CONN['LIPS'], C_LIPS, 2)
    return img


# ------------------------------------------------------------------ triptych
def run_ff(cmd, stdin_bytes=None):
    return subprocess.run(cmd, input=stdin_bytes, capture_output=True)


def make_triptych(gen, real, audio, lm, out_mp4, poster, labels, force=False):
    srcs = [gen, real, audio, lm]
    if (not force and os.path.exists(out_mp4) and os.path.exists(poster)
            and os.path.exists(out_mp4 + '.ver') and tload(out_mp4 + '.ver') == RENDER_VERSION + '|' + labels[0]
            and all(os.path.getmtime(out_mp4) > os.path.getmtime(p) for p in srcs)):
        return True
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    d = np.load(lm)
    face, pose = d['face'], d['pose']
    fok = d['face_ok'] if 'face_ok' in d else np.ones(len(face), bool)
    pok = d['pose_ok'] if 'pose_ok' in d else np.ones(len(pose), bool)
    T = 240
    frames = [draw_cond(face[min(i, len(face) - 1)], pose[min(i, len(pose) - 1)], bool(fok[min(i, len(fok) - 1)]), bool(pok[min(i, len(pok) - 1)])) for i in range(T)]
    raw = np.stack(frames).tobytes()
    ft = lambda text, x: (f"drawtext=fontfile={FONT}:text='{text}':x={x}:y=h-th-8:fontsize=15:fontcolor=white:"
                          f"box=1:boxcolor=black@0.6:boxborderw=4")
    fc = (f"[1:v]fps={FPS},scale={SIZE}:{SIZE}:flags=area,setsar=1[g];[2:v]fps={FPS},scale={SIZE}:{SIZE}:flags=area,setsar=1[r];"
          f"[0:v]setsar=1[c];[c][g][r]hstack=3,{ft('conditioning', 8)},{ft(labels[0], SIZE + 8)},{ft('real target', 2 * SIZE + 8)}[o]")
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{SIZE}x{SIZE}', '-r', str(FPS), '-i', '-',
           '-i', gen, '-i', real, '-i', audio, '-filter_complex', fc, '-map', '[o]', '-map', '3:a:0',
           '-t', '8.0', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(CRF), '-pix_fmt', 'yuv420p', '-r', str(FPS),
           '-c:a', 'aac', '-b:a', '80k', '-ac', '1', '-movflags', '+faststart', out_mp4]
    r = run_ff(cmd, raw)
    if r.returncode != 0:
        print('ffmpeg failed', out_mp4, r.stderr.decode()[-400:], file=sys.stderr); return False
    run_ff(['ffmpeg', '-v', 'error', '-y', '-i', out_mp4, '-frames:v', '1', '-q:v', '5', poster])
    with open(out_mp4 + '.ver', 'w') as f: f.write(RENDER_VERSION + '|' + labels[0])
    return os.path.exists(poster)


# ------------------------------------------------------------------ page sections
def fmt_cell(key, v, best=False):
    t = disp(key, v)
    return f'<b class="best">{t}</b>' if best and v is not None else t


def headline(rows, sc=None):
    br = best_run(rows)
    cols = []
    hl = [x for x in ((sc or {}).get('headline') or []) if x.get('run') in rows]
    if hl:
        for x in hl: cols.append((f'{esc(x.get("title") or x["run"])}<br><small>{esc(x.get("note") or "")}</small>', rows[x['run']]))
    elif br: cols.append((f'Best run<br><small>{esc(br["label"])}</small>', br))
    else: cols.append(('Best run<br><small>no runs yet</small>', None))
    nm = len(cols)
    cols.append(('Real video<br><small>ceiling</small>', rows.get('real')))
    cols.append(('Render → photoreal<br><small>fast model</small>', rows.get('render2real_fast')))
    body = []
    for key, al, hdr, lab, hib, nd in METRICS:
        if not lab: continue
        vals = [(c[1] or {}).get(key) for c in cols]
        cand = [v for i, v in enumerate(vals) if i != nm and v is not None]  # real is the ceiling, not a contender
        bestv = (max(cand) if hib else min(cand)) if cand else None
        tds = ''.join(f'<td>{fmt_cell(key, v, best=(i != nm and v is not None and v == bestv and len(cand) > 1))}</td>' for i, v in enumerate(vals))
        body.append(f'<tr><th scope="row">{esc(lab)}</th>{tds}</tr>')
    head = ''.join(f'<th>{h}</th>' for h, _ in cols)
    note = '' if br else '<p class="mut">No runs have been scored yet. This table fills in when the first run is evaluated.</p>'
    return (f'<div class="tw"><table class="hl"><thead><tr><th>Metric</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            f'<p class="mut small">Bold = best of the models. Real video is the ceiling, shown for scale.</p>{note}')


def ablation(rows):
    if not rows: return '<p class="mut">No runs yet.</p>'
    metric_keys = [m[0] for m in METRICS]
    nonref = [r for n, r in rows.items() if n not in REFS]
    best = {}
    for k in metric_keys:
        vals = [r[k] for r in nonref if r.get(k) is not None]
        if len(vals) > 0: best[k] = (max(vals) if MK[k][4] else min(vals)) if MK[k][4] is not None else None
    for k in ('offset',): best[k] = None
    if nonref:
        vals = [abs(r['offset']) for r in nonref if r.get('offset') is not None]
        if vals: best['offset_abs'] = min(vals)
    head = ['Run'] + [h for _, h in CONFIG_COLS] + [MK[k][2] for k in metric_keys]
    th = ''.join(('<th class="first">' if i == 0 else '<th>') + esc(h) + '</th>' for i, h in enumerate(head))
    ncol = len(head)
    groups = []
    for n in REFS:
        if n in rows: groups.append(('reference rows', [rows[n]]))
    order = []
    for n, r in rows.items():
        if n in REFS: continue
        if r['group'] not in order: order.append(r['group'])
    pref = ['best', 'lip-sync loss', 'injection', 'encoder', 'audio / lookahead', 'data', 'backbone', 'identity reference']
    order.sort(key=lambda g: (pref.index(g) if g in pref else 99, g))
    for g in order: groups.append((g, [r for n, r in rows.items() if n not in REFS and r['group'] == g]))
    body = []
    for g, rs in groups:
        body.append(f'<tr class="grp"><td colspan="{ncol}">{esc(g)}</td></tr>')
        for r in rs:
            ref = r['run'] in REFS
            tds = [f'<td class="first">{esc(r["label"])}<br><small class="mut">{esc(r["run"])}</small></td>']
            for k, _ in CONFIG_COLS:
                v = r['cfg'].get(k)
                tds.append(f'<td>{esc(v) if v is not None else "<span class=mut>—</span>"}</td>')
            for k in metric_keys:
                v = r.get(k)
                if k == 'offset':
                    isb = (not ref) and v is not None and abs(v) == best.get('offset_abs')
                else:
                    isb = (not ref) and v is not None and best.get(k) is not None and v == best[k]
                tds.append(f'<td class="{"bestc" if isb else ""}">{fmt_cell(k, v, isb)}</td>')
            body.append(f'<tr class="{"ref" if ref else ""}">{"".join(tds)}</tr>')
    return (f'<div class="tw wide"><table class="abl"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="mut small">Swipe sideways for more columns. Bold on blue = best run in that column (reference rows excluded). Offset: closest to 0 is best.</p>')


def spend_section(root):
    s = jload(os.path.join(root, 'pods', 'spend.json'))
    out = []
    if s is None: out.append('<p class="mut">No spend recorded yet.</p>')
    else:
        if isinstance(s, list): s = {'pods': s}
        pods = s.get('pods') or []
        usd = num(s.get('usd'))
        if usd is None and pods: usd = sum(num(p.get('usd')) or 0 for p in pods)
        cap = num(s.get('cap_usd'))
        if usd is not None:
            out.append(f'<p class="big">${usd:,.2f}' + (f' <span class="mut">of ${cap:,.0f} cap</span>' if cap else '') + '</p>')
            if cap: out.append(f'<div class="bar"><i style="width:{min(100, 100 * usd / cap):.0f}%"></i></div>')
        if pods:
            rows = ''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
                esc(p.get('name', '')), esc(p.get('gpu', '')),
                f'${num(p.get("rate")):.2f}/h' if num(p.get('rate')) is not None else '—',
                f'{num(p.get("hours")):.1f}' if num(p.get('hours')) is not None else '—',
                f'${num(p.get("usd")):.2f}' if num(p.get('usd')) is not None else '—') for p in pods)
            out.append(f'<div class="tw"><table><thead><tr><th>Pod</th><th>GPU</th><th>Rate</th><th>Hours</th><th>Cost</th></tr></thead><tbody>{rows}</tbody></table></div>')
    out.append('<p>Judge and all LLM calls: Claude subscription, $0 API.</p>')
    return '\n'.join(out)


def text_section(title, sid, text):
    h = md(text)
    if not h: return ''
    return f'<section id="{sid}"><h2>{title}</h2><div class="card">{h}</div></section>'


# ------------------------------------------------------------------ main
CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#f5f7fa;--mut:#a9b4c2;--blue:#6cb6ff;--orange:#ffb454;--bestbg:#12304f}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow-x:hidden}
main{max-width:960px;margin:0 auto;padding:14px 12px 60px}
h1{font-size:1.55rem;line-height:1.25;margin:.4em 0 .3em}
h2{font-size:1.25rem;margin:1.8em 0 .5em;border-bottom:2px solid var(--line);padding-bottom:.25em}
h3,h4,h5{margin:1em 0 .3em}h3{font-size:1.05rem}
a{color:var(--blue)}code{background:#222a35;padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.mut{color:var(--mut)}.small{font-size:.85rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}
.card p{margin:.5em 0}.card ul,.card ol{padding-left:1.2em;margin:.5em 0}
.sum{border-left:4px solid var(--orange)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:8px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}
thead th{background:#212936;position:sticky;top:0}
th[scope=row],td.first,th.first{text-align:left;white-space:normal;min-width:9em}
table.hl th[scope=row]{min-width:7em;font-size:.85rem;font-weight:500}table.hl td{text-align:center}table.hl thead th{text-align:center;white-space:normal;min-width:5em}
table.abl td.first,table.abl th.first{position:sticky;left:0;background:var(--panel);z-index:1;border-right:1px solid var(--line)}
table.abl thead th.first{background:#212936;z-index:2}
tr.grp td{background:#1d2530;color:var(--orange);font-weight:700;text-align:left;text-transform:uppercase;font-size:.8rem;letter-spacing:.04em}
tr.ref td{background:#1a2028}tr.ref td.first{background:#1a2028}
.best{color:var(--blue);font-weight:800}td.bestc{background:var(--bestbg);box-shadow:inset 0 0 0 1px var(--blue)}
.clip{margin:16px 0 6px;font-weight:700;color:var(--orange)}
.vid{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px;margin:8px 0}
.vid .cap{font-size:.9rem;margin:0 0 6px}.vid .cap b{color:var(--blue)}
video{width:100%;height:auto;aspect-ratio:3/1;background:#000;border-radius:6px;display:block}
.big{font-size:1.6rem;font-weight:700;margin:.2em 0}
.bar{height:10px;background:#222a35;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}
.legend span{display:inline-block;margin-right:12px}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px;margin-right:4px}
"""


def build(args):
    if args.dummy:
        root, pagedir, out = os.path.join(HERE, 'dummy'), os.path.join(HERE, 'dummy'), os.path.join(HERE, 'dummy_out')
        if not os.path.exists(os.path.join(root, 'DUMMY_READY')):
            subprocess.check_call([sys.executable, os.path.join(HERE, 'make_dummy.py')])
    else:
        root, pagedir, out = REAL_ROOT, HERE, args.out or ARTIFACT
    vdir = os.path.join(out, 'videos'); os.makedirs(vdir, exist_ok=True)

    rows = load_rows(root); normalise_percent(rows)
    order_runs = sorted([r for n, r in rows.items() if n not in REFS and r.get('lse_c') is not None], key=lambda r: -r['lse_c'])
    order_runs += [r for n, r in rows.items() if n not in REFS and r.get('lse_c') is None]

    # ---- showcase
    sc = jload(os.path.join(pagedir, 'showcase.json'), {}) or {}
    sruns = sc.get('runs') or [{'run': r['run'], 'title': r['label'], 'note': ''} for r in order_runs[:3]]
    sruns = [x if isinstance(x, dict) else {'run': x} for x in sruns]
    sruns = [x for x in sruns if os.path.isdir(os.path.join(root, 'runs', x['run'], 'out'))]
    man = jload(os.path.join(root, 'data', 'manifest.json'), {}) or {}
    held = [c['id'] for c in man.get('clips', []) if c.get('split') == 'heldout']
    cand = sc.get('clips') or held
    clips = []
    for c in cand:
        d = os.path.join(root, 'data', 'clips', c)
        if all(os.path.exists(os.path.join(d, f)) for f in ('real.mp4', 'audio.wav', 'lm.npz')): clips.append(c)
    if not sc.get('clips'): clips = clips[:4]

    vids_html = []
    nvid = 0
    if args.no_videos: pass
    for c in clips:
        d = os.path.join(root, 'data', 'clips', c)
        items = []
        for x in sruns:
            gen = os.path.join(root, 'runs', x['run'], 'out', c + '.mp4')
            if not os.path.exists(gen): continue
            name = f'{x["run"]}__{c}'
            mp4, jpg = os.path.join(vdir, name + '.mp4'), os.path.join(vdir, name + '.jpg')
            lab = (x.get('title') or x['run'])[:28]
            ok = True if args.no_videos and os.path.exists(mp4) else make_triptych(
                gen, os.path.join(d, 'real.mp4'), os.path.join(d, 'audio.wav'), os.path.join(d, 'lm.npz'), mp4, jpg,
                (re.sub(r"[':\\%]", '', lab),), force=args.force)
            if not ok: continue
            nvid += 1
            note = f' <span class="mut">{esc(x["note"])}</span>' if x.get('note') else ''
            items.append(f'<div class="vid"><p class="cap"><b>{esc(x.get("title") or x["run"])}</b>{note}</p>'
                         f'<video controls preload="none" playsinline poster="videos/{esc(name)}.jpg">'
                         f'<source src="videos/{esc(name)}.mp4" type="video/mp4"></video></div>')
        if items: vids_html.append(f'<div class="clip">{esc(c)}</div>' + ''.join(items))
    if vids_html:
        clipsec = ('<p class="mut small">Left to right: the skeleton and face landmarks fed to the model, what the model made, the real recording. '
                   'Tap play (sound on). For detail, go full screen and rotate the phone.</p>'
                   '<p class="legend small"><span><i class="sw" style="background:#5ae6ff"></i>face contour</span>'
                   '<span><i class="sw" style="background:#ffe63c"></i>lips</span>'
                   '<span><i class="sw" style="background:#ff8c00"></i>body skeleton</span></p>' + ''.join(vids_html))
    else:
        clipsec = '<p class="mut">No runs yet. Side-by-side clips appear here once a run has produced output for the held-out clips.</p>'

    # ---- sections
    summ = md(tload(os.path.join(pagedir, 'summary.md')))
    summary = f'<div class="card sum">{summ}</div>' if summ else '<div class="card sum"><p class="mut">Summary not written yet.</p></div>'
    nruns = len([n for n in rows if n not in REFS])
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>pose2real: landmarks + skeleton + audio to talking video</title>
<style>{CSS}</style></head><body><main>
<h1>pose2real: landmarks, skeleton and audio to a talking video</h1>
<p class="mut small">Held-out Seamless clips, 8 s, 384&times;384, 30 fps. {nruns} run{'s' if nruns != 1 else ''} scored. Built {time.strftime('%Y-%m-%d %H:%M')}.</p>
{summary}
<section id="headline"><h2>Headline</h2>{headline(rows, sc)}</section>
<section id="clips"><h2>Side by side, with sound</h2>{clipsec}</section>
<section id="ablation"><h2>All runs</h2>{ablation(rows)}</section>
{('<section id="extra">' + tload(os.path.join(pagedir, 'extra.html')) + '</section>') if os.path.exists(os.path.join(pagedir, 'extra.html')) else ''}
{text_section('What worked', 'worked', tload(os.path.join(pagedir, 'worked.md')))}
{text_section('What did not', 'didnt', tload(os.path.join(pagedir, 'didnt.md')))}
{text_section('Next', 'next', tload(os.path.join(pagedir, 'next.md')))}
<section id="spend"><h2>Spend</h2><div class="card">{spend_section(root)}</div></section>
</main></body></html>
"""
    with open(os.path.join(out, 'index.html'), 'w') as f: f.write(page)
    # drop videos no longer referenced
    ref = set(re.findall(r'videos/([^"]+)', page))
    for fn in os.listdir(vdir):
        base = fn[:-4] if fn.endswith('.ver') else fn
        if base not in ref:
            try: os.remove(os.path.join(vdir, fn))
            except OSError: pass
    print(f'wrote {out}/index.html  runs={nruns} clips={len(clips)} videos={nvid}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--no-videos', action='store_true')
    ap.add_argument('--force', action='store_true', help='re-encode all triptychs')
    build(ap.parse_args())
