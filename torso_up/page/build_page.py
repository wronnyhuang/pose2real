#!/usr/bin/env python3
"""torso_up pose2real-TORSO results page.

  python ROOT/page/build_page.py                # real build -> $OUT_DIR/pose2real-torso/{index.html,videos/}
  python ROOT/page/build_page.py --dummy        # build from page/dummy/ (made on demand by make_dummy.py) -> page/dummy_out/
  python ROOT/page/build_page.py --no-videos    # html only (reuse existing videos)
  python ROOT/page/build_page.py --force        # re-encode every video

Inputs (all optional; the page says "not yet" where something is missing):
  ROOT/eval/results/table.json     rows (list, {"rows":[...]} or dict of rows). Row = flat metrics (= mean over 20 clips) and/or
                                   nested {"mean":{..}, "mean_same12":{..}, "mean_gesture8":{..}}. Rows named bust_* = last round.
  ROOT/runs/<run>/run.json         group, label, ms_per_frame, gpu, train_minutes (+ optional note)
  ROOT/runs/<run>/out/<clip>.mp4   generated clips;  ROOT/data/clips/<clip>/{real.mp4,audio.wav,lm.npz,meta.json}
  ROOT/data/manifest.json, ROOT/pods/spend.json, ROOT/eval/README.md (optional: metric explanations)
  PAGE/{summary,worked,didnt,speed,next}.md, PAGE/showcase.json
Nothing is fetched from the network; the output folder is self-contained.
"""
import argparse, html, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_ROOT = os.path.dirname(HERE)
ARTIFACT = __import__('os').path.expandvars('$OUT_DIR/pose2real-torso')
BUST_DIR = __import__('os').path.expandvars('$OUT_DIR/pose2real/videos')   # read-only; files <run>__<clip>.mp4
BUST_BEST = 'v3_seen'
VENV_PY = __import__('os').path.expandvars('$WORK/eval/venv/bin/python')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
W, H, FPS, CRF = 512, 640, 30, 23
RENDER_VERSION = 'v2'
REFS = ('real', 'real_crf28')
REF_LABELS = {'real': 'Real video (ceiling)', 'real_crf28': 'Real video re-encoded (noise floor)'}
PANELS = ('Input: landmarks + skeleton + hands', 'Generated', 'Real video (never seen)')
PANEL_LINES = (('Input: landmarks', '+ skeleton + hands'), ('Generated', ''), ('Real video', '(never seen)'))

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
    """markdown-lite: paragraphs, bullets, numbered lists, #-headings, pipe tables, **bold**, *em*, `code`."""
    if not text or not text.strip(): return None
    out, lst, para, tbl = [], None, [], []
    def flush():
        nonlocal para
        if para: out.append('<p>' + inline(' '.join(para)) + '</p>'); para = []
    def close():
        nonlocal lst
        if lst: out.append(f'</{lst}>'); lst = None
    def flush_tbl():
        nonlocal tbl
        if not tbl: return
        rows = [[c.strip() for c in r.strip().strip('|').split('|')] for r in tbl]
        rows = [r for r in rows if not all(re.fullmatch(r':?-{2,}:?', c) for c in r)]
        if rows:
            hd = ''.join(f'<th>{inline(c)}</th>' for c in rows[0])
            bd = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows[1:])
            out.append(f'<div class="tw"><table class="md"><thead><tr>{hd}</tr></thead><tbody>{bd}</tbody></table></div>')
        tbl = []
    for ln in text.splitlines():
        s = ln.rstrip()
        if s.strip().startswith('|') and s.strip().count('|') >= 2:
            flush(); close(); tbl.append(s); continue
        flush_tbl()
        m = re.match(r'^(#{1,4})\s+(.*)', s)
        b = re.match(r'^\s*[-*]\s+(.*)', s); n = re.match(r'^\s*\d+[.)]\s+(.*)', s)
        if not s.strip(): flush(); close(); continue
        if m: flush(); close(); lv = min(len(m.group(1)) + 2, 5); out.append(f'<h{lv}>{inline(m.group(2))}</h{lv}>')
        elif b or n:
            flush(); kind = 'ul' if b else 'ol'
            if lst != kind: close(); out.append(f'<{kind}>'); lst = kind
            out.append('<li>' + inline((b or n).group(1)) + '</li>')
        else: close(); para.append(s.strip())
    flush(); close(); flush_tbl()
    return '\n'.join(out)


def num(v):
    if isinstance(v, bool): return None
    if isinstance(v, (int, float)) and v == v: return float(v)
    if isinstance(v, str):
        try: return float(v)
        except ValueError: return None
    return None


NEST = ('syncnet', 'landmarks', 'arm', 'hand', 'hands', 'realism', 'pose', 'sync', 'face', 'judge', 'extra')


def find(d, names):
    if not isinstance(d, dict): return None
    for n in names:
        if n in d and num(d[n]) is not None: return num(d[n])
    for k in NEST:
        v = d.get(k)
        if isinstance(v, dict):
            for n in names:
                if n in v and num(v[n]) is not None: return num(v[n])
    return None


# ------------------------------------------------------------------ metrics
# key, aliases, plain-words header, higher_is_better, kind ('n' number | 'p' rate shown as %), decimals, one-line meaning
METRICS = [
    ('lse_c', ('lse_c', 'LSE-C'), 'Lip-sync confidence', True, 'n', 2, 'SyncNet LSE-C: how confidently the lips match the audio. Real video scores about the ceiling value.'),
    ('lse_d', ('lse_d', 'LSE-D'), 'Lip-sync distance', False, 'n', 2, 'SyncNet LSE-D: mismatch between lips and audio.'),
    ('face_nme', ('face_nme',), 'Face error %', False, 'n', 2, 'Face landmarks re-measured on the output vs the input, as % of face size.'),
    ('mouth_nme', ('mouth_nme',), 'Mouth error %', False, 'n', 2, 'Same, mouth landmarks only.'),
    ('pose_err', ('pose_err',), 'Body-pose error', False, 'n', 2, 'Head and upper-body skeleton points, error as % of shoulder width.'),
    ('shoulder_err', ('shoulder_err',), 'Shoulder error', False, 'n', 2, 'Shoulders, % of shoulder width.'),
    ('elbow_err', ('elbow_err',), 'Elbow error', False, 'n', 2, 'Elbows, % of shoulder width.'),
    ('wrist_err', ('wrist_err',), 'Wrist error', False, 'n', 2, 'Wrists, % of shoulder width.'),
    ('arm_err', ('arm_err',), 'Arm error', False, 'n', 2, 'Elbows plus wrists together, % of shoulder width.'),
    ('hand_err', ('hand_err',), 'Hand position error', False, 'n', 2, 'All 21 points per hand re-tracked on the output vs the input, % of shoulder width (frames where both have the hand).'),
    ('hand_err_hs', ('hand_err_hs',), 'Hand error (hand-size units)', False, 'n', 2, 'Same, but as % of the hand\'s own size.'),
    ('hand_det_rate', ('hand_det_rate',), 'Hands found in output', True, 'p', 0, 'Share of frames where the input has a hand and the tracker finds a hand in the output too.'),
    ('hand_extra_rate', ('hand_extra_rate',), 'Phantom hands', False, 'p', 0, 'Share of frames where the output shows a hand that the input did not have.'),
    ('finger_pose_err', ('finger_pose_err',), 'Finger shape error', False, 'n', 2, 'Hand shape only (position, size and rotation removed), % of hand size.'),
    ('finger_angle_err_deg', ('finger_angle_err_deg',), 'Finger angle error (deg)', False, 'n', 1, 'Average finger joint angle difference, degrees.'),
    ('realism', ('realism', 'score_1_10', 'realism_score'), 'Realism (1-10)', True, 'n', 1, 'AI judge score for how real the whole clip looks.'),
    ('hands_score', ('hands_score_1_10', 'hands_score'), 'Hand realism (1-10)', True, 'n', 1, 'AI judge score looking only at hand close-ups.'),
    ('fooled_rate', ('fooled_rate',), 'Judge fooled', True, 'p', 0, 'Share of trials where the judge called the generated clip real.'),
    ('offset', ('offset', 'offset_frames'), 'Sync offset (frames)', None, 'n', 1, 'Audio-to-lip delay in frames; closest to 0 is best.'),
]
MK = {m[0]: m for m in METRICS}
CFG_COLS = [('gpu', 'GPU'), ('train_minutes', 'Train minutes'), ('ms_per_frame', 'Speed (ms per frame, lower is better)')]


def metric_value(raw, key, subset=None):
    """subset None -> the 20-clip mean (flat keys or raw['mean']); 'same12' -> raw['mean_same12'] / flat key_same12 only."""
    al = MK[key][1]
    if subset:
        for src in (raw.get('mean_' + subset), raw.get(subset)):
            v = find(src, al)
            if v is not None: return v
        return find(raw, tuple(a + '_' + subset for a in al))
    v = find(raw.get('mean'), al) if isinstance(raw.get('mean'), dict) else None
    return v if v is not None else find(raw, al)


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
    rd = os.path.join(root, 'runs')
    rows = {}
    for name, r in items:
        row = {'run': name, 'raw': r}
        row['m'] = {k: metric_value(r, k) for k in MK}
        row['s12'] = {k: metric_value(r, k, 'same12') for k in MK}
        rows[name] = row
    if os.path.isdir(rd):   # runs known only from runs/<run>/run.json
        for n in sorted(os.listdir(rd)):
            if os.path.exists(os.path.join(rd, n, 'run.json')) and n not in rows:
                rows[n] = {'run': n, 'raw': {}, 'm': {k: None for k in MK}, 's12': {k: None for k in MK}}
    for name, row in rows.items():
        cfg = jload(os.path.join(rd, name, 'run.json'), {}) or {}
        row['cfg'] = cfg if isinstance(cfg, dict) else {}
        if row['m'].get('offset') is None: pass
        ms = find(row['cfg'], ('ms_per_frame',))
        if ms is None: ms = find(row['raw'], ('ms_per_frame',))
        row['ms'] = ms
        row['bust'] = name.startswith('bust_')
        row['ref'] = name in REFS
        row['group'] = 'last round, bust framing' if row['bust'] else ('reference rows' if row['ref'] else str(row['cfg'].get('group') or 'other'))
        row['label'] = REF_LABELS.get(name) or str(row['cfg'].get('label') or row['raw'].get('label') or (name[5:] + ' (bust)' if name.startswith('bust_') else name))
        row['has_out'] = os.path.isdir(os.path.join(rd, name, 'out'))
    return rows


def disp(key, v):
    if v is None: return '<span class="mut">–</span>'
    kind, nd = MK[key][4], MK[key][5]
    return f'{v * 100:.0f}%' if kind == 'p' else f'{v:.{nd}f}'


def fmt(v, nd=0): return '<span class="mut">–</span>' if v is None else f'{v:.{nd}f}'


def best_torso(rows):
    cand = [r for r in rows.values() if not r['bust'] and not r['ref'] and r['m'].get('lse_c') is not None]
    return max(cand, key=lambda r: r['m']['lse_c']) if cand else None


# ------------------------------------------------------------------ conditioning panel
CONN = jload(os.path.join(HERE, 'connections.json'))
HAND_BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
              (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
ARM_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (7, 11), (8, 12)]
# BGR
C_FACE, C_FACE_LINE, C_LIPS, C_POSE = (160, 150, 140), (120, 105, 95), (60, 230, 255), (255, 255, 255)
C_HAND = [(84, 180, 255), (255, 182, 108)]   # left hand orange, right hand blue (RGB #ffb454 / #6cb6ff)
HAND_NAMES = ['left hand', 'right hand']


def P(p, w=W, h=H):
    return (int(round(float(np.clip(p[0], -3, 4)) * w)), int(round(float(np.clip(p[1], -3, 4)) * h)))


def draw_cond(face, pose, hands, face_ok, pose_ok, hands_ok):
    img = np.zeros((H, W, 3), np.uint8)
    if face_ok:
        for k in ('FACE_OVAL', 'LEFT_EYE', 'RIGHT_EYE', 'LEFT_EYEBROW', 'RIGHT_EYEBROW'):
            for a, b in CONN[k]: cv2.line(img, P(face[a]), P(face[b]), C_FACE_LINE, 1, cv2.LINE_AA)
        for p in face: cv2.circle(img, P(p), 1, C_FACE, -1)
        for a, b in CONN['LIPS']: cv2.line(img, P(face[a]), P(face[b]), C_LIPS, 2, cv2.LINE_AA)
    if pose_ok:
        vis = pose[:, 3] > 0.5
        for a, b in ARM_BONES:
            if vis[a] and vis[b]: cv2.line(img, P(pose[a]), P(pose[b]), C_POSE, 4, cv2.LINE_AA)
        for i in (11, 12, 13, 14, 15, 16, 23, 24):
            if vis[i]: cv2.circle(img, P(pose[i]), 6, C_POSE, -1, cv2.LINE_AA)
    for h in range(2):
        if hands is not None and hands_ok[h]:
            pts = hands[h]
            for a, b in HAND_BONES: cv2.line(img, P(pts[a]), P(pts[b]), C_HAND[h], 2, cv2.LINE_AA)
            for i, p in enumerate(pts): cv2.circle(img, P(p), 4 if i in (4, 8, 12, 16, 20) else 3, C_HAND[h], -1, cv2.LINE_AA)
    return img


def load_lm(path):
    d = np.load(path)
    face, pose = d['face'], d['pose']
    T = len(face)
    fok = d['face_ok'] if 'face_ok' in d else np.ones(T, bool)
    pok = d['pose_ok'] if 'pose_ok' in d else np.ones(len(pose), bool)
    hands = d['hands'] if 'hands' in d else None
    hok = d['hands_ok'] if 'hands_ok' in d else (np.ones((T, 2), bool) if hands is not None else None)
    return face, pose, hands, fok, pok, hok


# ------------------------------------------------------------------ videos
def ver_ok(path, ver, srcs):
    return (os.path.exists(path) and os.path.exists(path + '.ver') and tload(path + '.ver') == ver
            and all(os.path.exists(p) and os.path.getmtime(path) > os.path.getmtime(p) for p in srcs))


def make_triptych(gen, real, audio, lm, out_mp4, poster, force=False):
    srcs = [gen, real, audio, lm]
    if not force and ver_ok(out_mp4, RENDER_VERSION, srcs) and os.path.exists(poster): return True
    os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
    face, pose, hands, fok, pok, hok = load_lm(lm)
    def at(a, i): return a[min(i, len(a) - 1)]
    legend = [('lips', C_LIPS), ('left hand', C_HAND[0]), ('right hand', C_HAND[1]), ('body', C_POSE)]
    frames = []
    for i in range(240):
        img = draw_cond(at(face, i), at(pose, i), at(hands, i) if hands is not None else None, bool(at(fok, i)), bool(at(pok, i)),
                        at(hok, i) if hok is not None else [False, False])
        x = 8
        for t, c in legend:
            cv2.circle(img, (x + 7, H - 20), 7, c, -1, cv2.LINE_AA)
            cv2.putText(img, t, (x + 18, H - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 240, 240), 1, cv2.LINE_AA)
            x += 34 + 11 * len(t)
        frames.append(img)
    raw = np.stack(frames).tobytes()
    def dt(text, x, y):
        if not text: return None
        text = text.replace(':', '\\:')
        return (f"drawtext=fontfile={FONT}:text='{text}':x={x}+({W}-text_w)/2:y={y}:fontsize=34:fontcolor=white:"
                f"box=1:boxcolor=black@0.8:boxborderw=6")
    norm = lambda i, n: f"[{i}:v]fps={FPS},scale={W}:{H}:flags=area,setsar=1[{n}]"
    dts = [dt(t, k * W, 8 + 46 * j) for k, ln in enumerate(PANEL_LINES) for j, t in enumerate(ln)]
    fc = (f"{norm(1, 'g')};{norm(2, 'r')};[0:v]setsar=1[c];[c][g][r]hstack=3," + ','.join(d for d in dts if d) + "[o]")
    cmd = ['nice', '-n', '10', 'ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', gen, '-i', real, '-i', audio, '-filter_complex', fc, '-map', '[o]', '-map', '3:a:0', '-t', '8.0',
           '-threads', '2', '-c:v', 'libx264', '-preset', 'medium', '-crf', str(CRF), '-pix_fmt', 'yuv420p', '-r', str(FPS),
           '-c:a', 'aac', '-b:a', '96k', '-ac', '1', '-movflags', '+faststart', out_mp4]
    r = subprocess.run(cmd, input=raw, capture_output=True)
    if r.returncode != 0:
        print('ffmpeg failed', out_mp4, r.stderr.decode()[-500:], file=sys.stderr); return False
    subprocess.run(['nice', '-n', '10', 'ffmpeg', '-v', 'error', '-y', '-ss', '1.0', '-i', out_mp4, '-frames:v', '1', '-q:v', '4', poster])
    with open(out_mp4 + '.ver', 'w') as f: f.write(RENDER_VERSION)
    return os.path.exists(poster)


def read_frames(path, idx):
    want, out = set(idx), {}
    cap = cv2.VideoCapture(path); i = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if i in want: out[i] = f
        i += 1
        if i > max(want): break
    cap.release()
    return out


def make_hand_strip(gen, real, lm, out_jpg, force=False):
    """2x zoomed hand crops: row 1 generated, row 2 real, 4 frames where the input has a hand. Returns caption or None."""
    srcs = [gen, real, lm]
    if not force and ver_ok(out_jpg, RENDER_VERSION, srcs): return tload(out_jpg + '.ver.cap') or ''
    _, _, hands, _, _, hok = load_lm(lm)
    if hands is None: return None
    ok = [(t, h) for t in range(len(hands)) for h in range(2) if hok[t, h]]
    ts = sorted({t for t, _ in ok})
    if len(ts) < 2: return None
    pick = [ts[int(round(q * (len(ts) - 1)))] for q in (0.08, 0.36, 0.64, 0.92)]
    pick = sorted(set(pick))
    gf, rf = read_frames(gen, pick), read_frames(real, pick)
    S, C = 240, 120
    tiles = [[], []]
    for t in pick:
        hs = [h for h in range(2) if hok[t, h]]
        h = max(hs, key=lambda k: float(np.ptp(hands[t, k, :, 1])))   # the more spread hand
        cx, cy = hands[t, h, :, 0].mean() * W, hands[t, h, :, 1].mean() * H
        x0 = int(np.clip(cx - C / 2, 0, W - C)); y0 = int(np.clip(cy - C / 2, 0, H - C))
        for row, fr in enumerate((gf.get(t), rf.get(t))):
            if fr is None: fr = np.zeros((H, W, 3), np.uint8)
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA) if fr.shape[:2] != (H, W) else fr
            tile = cv2.resize(fr[y0:y0 + C, x0:x0 + C], (S, S), interpolation=cv2.INTER_CUBIC)
            lab = ('Gen' if row == 0 else 'Real') + f' {t / FPS:.1f}s ' + ('L' if h == 0 else 'R')
            cv2.rectangle(tile, (0, 0), (S, 34), (0, 0, 0), -1)
            cv2.putText(tile, lab, (6, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (S - 1, S - 1), (84, 180, 255) if row == 0 else (255, 182, 108), 2)
            tiles[row].append(tile)
    img = np.vstack([np.hstack(r) for r in tiles])
    cv2.imwrite(out_jpg, img, [cv2.IMWRITE_JPEG_QUALITY, 84])
    cap = 'Hand close-up, 2x zoom: top row generated (orange frame), bottom row real (blue frame). L = left hand, R = right hand.'
    with open(out_jpg + '.ver', 'w') as f: f.write(RENDER_VERSION)
    with open(out_jpg + '.ver.cap', 'w') as f: f.write(cap)
    return cap


# ------------------------------------------------------------------ page sections
def headline(rows):
    br = best_torso(rows)
    bust = rows.get('bust_' + BUST_BEST)
    if bust is None:
        bl = [r for r in rows.values() if r['bust'] and r['m'].get('lse_c') is not None]
        bust = max(bl, key=lambda r: r['m']['lse_c']) if bl else None
    cols = [('Best torso', f'{esc(br["label"])}' if br else 'none scored yet', br['s12'] if br else None, br['m'] if br else None),
            ('Real video', 'the ceiling', rows['real']['m'] if 'real' in rows else None, rows['real']['m'] if 'real' in rows else None),
            ('Best bust (last round)', esc(bust['label']) if bust else 'not in table', bust['m'] if bust else None, None)]
    # torso column uses the 20-clip mean (all runs); same12 appears in the bust section
    cols[0] = (cols[0][0], cols[0][1], cols[0][3], None)
    keys = ['lse_c', 'lse_d', 'face_nme', 'mouth_nme', 'pose_err', 'arm_err', 'hand_err', 'hand_det_rate', 'finger_pose_err', 'realism', 'hands_score', 'fooled_rate']
    body = []
    for k in keys:
        m = MK[k]
        vals = [(c[2] or {}).get(k) for c in cols]
        cand = [vals[0]] if vals[0] is not None else []
        if vals[2] is not None and k not in ('hand_err', 'hand_det_rate', 'finger_pose_err', 'hands_score', 'arm_err'): cand.append(vals[2])
        bv = (max(cand) if m[3] else min(cand)) if len(cand) > 1 else None
        tds = []
        for i, v in enumerate(vals):
            b = (i != 1) and v is not None and bv is not None and v == bv
            t = disp(k, v)
            tds.append(f'<td>{"<b class=best>" + t + "</b>" if b else t}</td>')
        arrow = 'higher = better' if m[3] else 'lower = better'
        body.append(f'<tr><th scope="row">{esc(m[2])}<br><small class="mut">{arrow}</small></th>{"".join(tds)}</tr>')
    head = ''.join(f'<th>{h}<br><small class="mut">{s}</small></th>' for h, s, _, _ in cols)
    note = '' if br else '<p class="mut">No torso run has been scored yet; this table fills in when the first one is evaluated.</p>'
    return (f'<div class="tw"><table class="hl"><thead><tr><th>Number</th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="mut small">Torso and real columns: mean over the 20 held-out clips. Bust column: last round\'s best, 12 clips. '
            'Bold blue = better of torso vs bust where the number is comparable. "–" = not measured (the bust round had no hands or arms scoring). '
            'Real video is the ceiling, shown for scale. Face errors are not strictly comparable across framings (the face is smaller in torso framing).</p>' + note)


def ablation(rows):
    runs = [r for r in rows.values() if not r['bust']]
    if not runs: return '<p class="mut">No runs yet.</p>'
    mkeys = [m[0] for m in METRICS]
    nonref = [r for r in runs if not r['ref']]
    best = {}
    for k in mkeys:
        vals = [r['m'][k] for r in nonref if r['m'].get(k) is not None]
        if k == 'offset': vals = [abs(v) for v in vals]
        if vals: best[k] = max(vals) if MK[k][3] else min(vals)
    mss = [r['ms'] for r in nonref if r['ms'] is not None]
    if mss: best['ms'] = min(mss)
    head = ['Run'] + [h for _, h in CFG_COLS]
    th = '<th class="first">Run</th>' + ''.join(f'<th>{esc(h)}</th>' for h in head[1:])
    for k in mkeys:
        m = MK[k]
        a = '' if m[3] is None else ('higher is better' if m[3] else 'lower is better')
        th += f'<th>{esc(m[2])}<br><small class="mut">{a if a else "closest to 0 is best"}</small></th>'
    ncol = 1 + len(CFG_COLS) + len(mkeys)
    pref = ['best', 'lip-sync loss', 'injection', 'encoder', 'audio / lookahead', 'data', 'backbone', 'identity reference']
    order = []
    for r in nonref:
        if r['group'] not in order: order.append(r['group'])
    order.sort(key=lambda g: (pref.index(g) if g in pref else 99, g))
    groups = []
    refs = [rows[n] for n in REFS if n in rows]
    if refs: groups.append(('reference rows (pinned)', refs))
    for g in order: groups.append((g, [r for r in nonref if r['group'] == g]))
    body = []
    for g, rs in groups:
        body.append(f'<tr class="grp"><td colspan="{ncol}"><div class="gl">{esc(g)}</div></td></tr>')
        for r in rs:
            tds = [f'<td class="first">{esc(r["label"])}<br><small class="mut">{esc(r["run"])}</small></td>']
            tds.append(f'<td>{esc(r["cfg"].get("gpu")) if r["cfg"].get("gpu") else "<span class=mut>–</span>"}</td>')
            tm = num(r['cfg'].get('train_minutes'))
            tds.append(f'<td>{fmt(tm, 0)}</td>')
            isb = (not r['ref']) and r['ms'] is not None and r['ms'] == best.get('ms')
            tds.append(f'<td class="{"bestc" if isb else ""}">{fmt(r["ms"], 0)}</td>')
            for k in mkeys:
                v = r['m'].get(k)
                cmp = abs(v) if (k == 'offset' and v is not None) else v
                isb = (not r['ref']) and v is not None and best.get(k) is not None and cmp == best[k] and len(nonref) > 1
                t = disp(k, v)
                tds.append(f'<td class="{"bestc" if isb else ""}">{"<b class=best>" + t + "</b>" if isb else t}</td>')
            body.append(f'<tr class="{"ref" if r["ref"] else ""}">{"".join(tds)}</tr>')
    return (f'<div class="tw wide"><table class="abl"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
            '<p class="mut small">Swipe sideways for more columns; the run name stays pinned. Bold on blue = best run in that column '
            '(real-video rows excluded from the contest). Every number is the mean over the 20 held-out clips. "–" = not measured.</p>')


def how_to_read(root):
    expl = {m[0]: m[6] for m in METRICS}
    rd = tload(os.path.join(root, 'eval', 'README.md')) or ''
    for ln in rd.splitlines():
        mm = re.match(r'^\s*[-*|]\s*\**`?([a-z_0-9]+)`?\**\s*[:|—–-]+\s*(.+?)\s*\|?\s*$', ln)
        if mm and mm.group(1) in expl and len(mm.group(2)) > 8: expl[mm.group(1)] = mm.group(2)
    items = ''.join(f'<li><b>{esc(m[2])}</b>: {inline(expl[m[0]])}</li>' for m in METRICS)
    return f'<details class="card"><summary><b>How to read these numbers</b> (tap)</summary><ul>{items}</ul></details>'


def bust_section(rows, root, vdir, jobs, clips_pref, args, brun):
    bust_rows = [r for r in rows.values() if r['bust']]
    torso = [r for r in rows.values() if not r['bust'] and not r['ref'] and r['s12'].get('lse_c') is not None]
    ref = rows.get('real')
    keys = ['lse_c', 'lse_d', 'face_nme', 'mouth_nme', 'pose_err', 'realism', 'fooled_rate', 'arm_err', 'hand_err', 'hand_det_rate']
    if not bust_rows and not torso: return '<p class="mut">Not available yet.</p>'
    th = '<th class="first">Run</th>' + ''.join(f'<th>{esc(MK[k][2])}<br><small class="mut">{"higher" if MK[k][3] else "lower"} is better</small></th>' for k in keys)
    def line(r, src, cls=''):
        tds = ''.join(f'<td>{disp(k, r[src].get(k))}</td>' for k in keys)
        return f'<tr class="{cls}"><td class="first">{esc(r["label"])}<br><small class="mut">{esc(r["run"])}</small></td>{tds}</tr>'
    body = [f'<tr class="grp"><td colspan="{len(keys) + 1}"><div class="gl">last round: bust framing, same 12 windows</div></td></tr>']
    body += [line(r, 'm') for r in sorted(bust_rows, key=lambda r: -(r['m'].get('lse_c') or -1))]
    body.append(f'<tr class="grp"><td colspan="{len(keys) + 1}"><div class="gl">this round: torso framing, same 12 windows</div></td></tr>')
    body += [line(r, 's12') for r in sorted(torso, key=lambda r: -(r['s12'].get('lse_c') or -1))]
    if not torso: body.append(f'<tr><td class="first mut" colspan="{len(keys) + 1}">no torso run scored yet</td></tr>')
    tbl = f'<div class="tw wide"><table class="abl"><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'
    # videos: same clip, bust best vs torso best
    vids = []
    if brun is not None:
        for c in clips_pref:
            src = os.path.join(BUST_DIR, f'{BUST_BEST}__{c}.mp4')
            if not os.path.exists(src): continue
            name = f'bust_{BUST_BEST}__{c}'
            dst = os.path.join(vdir, name + '.mp4')
            if not args.no_videos:
                for ext in ('.mp4', '.jpg'):
                    s = src[:-4] + ext
                    if os.path.exists(s) and (not os.path.exists(dst[:-4] + ext) or os.path.getmtime(dst[:-4] + ext) < os.path.getmtime(s)):
                        shutil.copy2(s, dst[:-4] + ext)
            tname = f'{brun["run"]}__{c}'
            if not jobs_has(jobs, tname):
                add_job(jobs, root, brun['run'], c, vdir, force=args.force)
            if not os.path.exists(dst): continue
            vids.append(f'<div class="clip">{esc(c)}</div>'
                        f'<div class="vid"><p class="cap"><b>Bust framing</b> (last round, best run {BUST_BEST}). Left to right: input landmarks, generated, real.</p>'
                        + video_tag(name, with_poster=os.path.exists(dst[:-4] + '.jpg')) + '</div>'
                        f'<div class="vid"><p class="cap"><b>Torso framing</b> (this round, {esc(brun["label"])}). Panels: input landmarks + skeleton + hands, generated, real.</p>'
                        + video_tag(tname) + '</div>')
    vtxt = ''.join(vids) or '<p class="mut">No side-by-side videos yet.</p>'
    return tbl + ('<p class="mut small">Same 12 held-out windows in both rounds (torso re-crops the same footage). The bust round had no hand or arm scoring, so those cells are empty. '
                  'Face errors are not strictly comparable across framings.</p>') + vtxt


def video_tag(name, with_poster=True):
    p = f' poster="videos/{esc(name)}.jpg"' if with_poster else ''
    return (f'<video controls preload="none" playsinline{p} onerror="vfail(this)"><source src="videos/{esc(name)}.mp4" type="video/mp4"></video>'
            '<p class="vmsg mut small" hidden>This video could not be loaded.</p>')


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
            rws = ''.join('<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td class="note">{}</td></tr>'.format(
                esc(p.get('name', '')), esc(p.get('gpu', '')),
                f'${num(p.get("rate")):.2f}/h' if num(p.get('rate')) is not None else '–',
                f'{num(p.get("hours")):.1f}' if num(p.get('hours')) is not None else '–',
                f'${num(p.get("usd")):.2f}' if num(p.get('usd')) is not None else '–',
                esc(('terminated. ' if p.get('terminated') else 'still running. ') + (p.get('note') or ''))) for p in pods)
            out.append(f'<div class="tw"><table><thead><tr><th>Pod</th><th>GPU</th><th>Rate</th><th>Hours</th><th>Cost</th><th>Status</th></tr></thead><tbody>{rws}</tbody></table></div>')
    out.append('<p>The AI judge and every other model call ran on the Claude subscription: $0 API.</p>')
    return '\n'.join(out)


def text_section(title, sid, text):
    h = md(text)
    if not h: return ''
    return f'<section id="{sid}"><h2>{title}</h2><div class="card">{h}</div></section>'


# ------------------------------------------------------------------ jobs
def jobs_has(jobs, name): return name in jobs


def add_job(jobs, root, run, c, vdir, force=False):
    d = os.path.join(root, 'data', 'clips', c)
    gen = os.path.join(root, 'runs', run, 'out', c + '.mp4')
    if not os.path.exists(gen) or not all(os.path.exists(os.path.join(d, f)) for f in ('real.mp4', 'audio.wav', 'lm.npz')): return None
    name = f'{run}__{c}'
    jobs[name] = (gen, os.path.join(d, 'real.mp4'), os.path.join(d, 'audio.wav'), os.path.join(d, 'lm.npz'),
                  os.path.join(vdir, name + '.mp4'), os.path.join(vdir, name + '.jpg'), os.path.join(vdir, name + '_hands.jpg'), force)
    return name


def run_job(j):
    gen, real, audio, lm, mp4, jpg, hjpg, force = j
    ok = make_triptych(gen, real, audio, lm, mp4, jpg, force)
    cap = make_hand_strip(gen, real, lm, hjpg, force) if ok else None
    if ok and cap is None and os.path.exists(hjpg):
        for p in (hjpg, hjpg + '.ver', hjpg + '.ver.cap'):
            if os.path.exists(p): os.remove(p)
    return ok, cap


CSS = """
:root{--bg:#0d1117;--panel:#161b22;--line:#3a424d;--fg:#f5f7fa;--mut:#b4bfcc;--blue:#6cb6ff;--orange:#ffb454;--bestbg:#12304f}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;overflow-x:hidden}
main{max-width:1000px;margin:0 auto;padding:14px 12px 60px}
h1{font-size:1.5rem;line-height:1.25;margin:.4em 0 .3em}
h2{font-size:1.25rem;margin:1.8em 0 .5em;border-bottom:2px solid var(--line);padding-bottom:.25em}
h3,h4,h5{margin:1em 0 .3em}h3{font-size:1.05rem}
a{color:var(--blue)}code{background:#222a35;padding:1px 5px;border-radius:4px;overflow-wrap:anywhere}
.mut{color:var(--mut)}.small{font-size:.85rem}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:10px 0}
.card p{margin:.5em 0}.card ul,.card ol{padding-left:1.2em;margin:.5em 0}
details.card summary{cursor:pointer;padding:4px 0}
.sum{border-left:4px solid var(--orange)}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;border:1px solid var(--line);border-radius:8px;margin:8px 0}
table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:right;vertical-align:top;white-space:nowrap}
td.note{white-space:normal;text-align:left;min-width:12em}
table.md td,table.md th{white-space:normal;text-align:left}
thead th{background:#212936;vertical-align:bottom}
th[scope=row]{text-align:left;white-space:normal;min-width:9em}
table.hl{font-size:.8rem}table.hl th,table.hl td{padding:6px 5px}table.hl td{text-align:center}table.hl thead th{text-align:center;white-space:normal;min-width:4.6em}table.hl th[scope=row]{min-width:5.5em;max-width:7.5em;font-weight:600}
table.abl td.first,table.abl th.first{position:sticky;left:0;background:var(--panel);z-index:1;border-right:2px solid var(--line);text-align:left;white-space:normal;min-width:8em;max-width:9em}
table.abl thead th.first{background:#212936;z-index:2}
table.abl thead th{white-space:normal;min-width:6.5em}
tr.grp td{background:#1d2530;color:var(--orange);font-weight:700;text-align:left;text-transform:uppercase;font-size:.78rem;letter-spacing:.04em;position:sticky;left:0}
div.gl{position:sticky;left:8px;width:calc(100vw - 60px);max-width:900px;white-space:normal}
tr.ref td{background:#1a2028}tr.ref td.first{background:#1a2028}
.best{color:var(--blue);font-weight:800}td.bestc{background:var(--bestbg);box-shadow:inset 0 0 0 1px var(--blue)}
.clip{margin:16px 0 6px;font-weight:700;color:var(--orange)}
.run{margin-top:22px}.run h3{color:var(--blue);margin-bottom:.1em}
.vid{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:8px;margin:8px 0}
.vid .cap{font-size:.9rem;margin:0 0 6px}.vid .cap b{color:var(--blue)}
video{width:100%;height:auto;aspect-ratio:1536/640;background:#000;border-radius:6px;display:block}
.vid img{width:100%;height:auto;border-radius:6px;display:block;margin-top:8px}
.big{font-size:1.6rem;font-weight:700;margin:.2em 0}
.bar{height:10px;background:#222a35;border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--blue)}
.legend span{display:inline-block;margin-right:12px}.sw{display:inline-block;width:12px;height:12px;border-radius:50%;vertical-align:-1px;margin-right:4px}
#err{display:none;background:#5a3a00;border:2px solid var(--orange);color:#fff;padding:10px;border-radius:8px;margin:8px 0}
"""

JS = """
window.vfail=function(v){var m=v.parentNode.querySelector('.vmsg');if(m)m.hidden=false;};
window.onerror=function(msg){var e=document.getElementById('err');if(e){e.style.display='block';e.textContent='Part of this page failed to run ('+msg+'). The numbers and text still show; videos can be opened with the play controls.';}};
"""


def sort_showcase(sc, rows):
    if isinstance(sc, dict): sc = sc.get('runs')
    return [x if isinstance(x, dict) else {'run': x} for x in (sc or [])]


def default_showcase(root, rows):
    man = jload(os.path.join(root, 'data', 'manifest.json'), {}) or {}
    cl = [c for c in man.get('clips', []) if c.get('split') == 'heldout']
    def top(s, n): return [c['id'] for c in sorted([c for c in cl if c.get('set') == s], key=lambda c: -(c.get('hand_frac') or 0))[:n]]
    clips = top('same12', 2) + top('gesture8', 2)
    if not clips: clips = [c['id'] for c in cl[:4]]
    runs = [r for r in rows.values() if not r['bust'] and not r['ref'] and r['has_out']]
    runs.sort(key=lambda r: -(r['m'].get('lse_c') or -1))
    return [{'run': r['run'], 'label': r['label'], 'clips': clips} for r in runs]


def build(args):
    try: os.nice(10)
    except OSError: pass
    if args.dummy:
        root = pagedir = os.path.join(HERE, 'dummy'); out = os.path.join(HERE, 'dummy_out')
        if not os.path.exists(os.path.join(root, 'DUMMY_READY')):
            subprocess.check_call([sys.executable, os.path.join(HERE, 'make_dummy.py')])
    else:
        root, pagedir, out = REAL_ROOT, HERE, args.out or ARTIFACT
    vdir = os.path.join(out, 'videos'); os.makedirs(vdir, exist_ok=True)
    rows = load_rows(root)
    sc = jload(os.path.join(pagedir, 'showcase.json'))
    show = sort_showcase(sc, rows) if sc else default_showcase(root, rows)
    man = jload(os.path.join(root, 'data', 'manifest.json'), {}) or {}
    cinfo = {c['id']: c for c in man.get('clips', [])}
    brun = best_torso(rows)

    jobs = {}
    plan = []   # (run entry, [(clip, name)])
    for x in show:
        if not os.path.isdir(os.path.join(root, 'runs', x['run'], 'out')): continue
        items = []
        for c in x.get('clips') or []:
            n = add_job(jobs, root, x['run'], c, vdir, args.force)
            if n: items.append((c, n))
        if items: plan.append((x, items))
    clips_pref = []
    for _, items in plan:
        for c, _ in items:
            if c not in clips_pref: clips_pref.append(c)
    same12 = [c['id'] for c in man.get('clips', []) if c.get('split') == 'heldout' and c.get('set') == 'same12']
    cmp_clips = [c for c in clips_pref if c in same12 or not same12]
    cmp_clips += [c for c in same12 if c not in cmp_clips]
    cmp_clips = [c for c in cmp_clips if os.path.exists(os.path.join(BUST_DIR, f'{BUST_BEST}__{c}.mp4'))][:4]

    bust_html = bust_section(rows, root, vdir, jobs, cmp_clips, args, brun)   # may add jobs
    res = {}
    if not args.no_videos:
        names = list(jobs)
        with ThreadPoolExecutor(4) as ex:
            for n, r in zip(names, ex.map(run_job, [jobs[n] for n in names])): res[n] = r
    else:
        for n, j in jobs.items(): res[n] = (os.path.exists(j[4]), tload(j[6] + '.ver.cap') if os.path.exists(j[6]) else None)

    # ---- clips
    blocks, nvid = [], 0
    for x, items in plan:
        b = [f'<div class="run"><h3>{esc(x.get("label") or x["run"])}</h3>']
        if x.get('note'): b.append(f'<p class="mut small">{esc(x["note"])}</p>')
        for c, n in items:
            ok, cap = res.get(n, (False, None))
            if not ok: continue
            nvid += 1
            ci = cinfo.get(c, {})
            info = ' &middot; '.join(esc(t) for t in (ci.get('set'), f'hands visible in {ci["hand_frac"]*100:.0f}% of frames' if ci.get('hand_frac') is not None else None) if t)
            strip = ''
            if cap is not None and os.path.exists(os.path.join(vdir, n + '_hands.jpg')):
                strip = f'<img src="videos/{esc(n)}_hands.jpg" alt="hand close-up" loading="lazy"><p class="mut small">{esc(cap)}</p>'
            b.append(f'<div class="clip">{esc(c)} <span class="mut small">{info}</span></div>'
                     f'<div class="vid"><p class="cap"><b>{esc(x.get("label") or x["run"])}</b> &mdash; panels: <b>Input: landmarks + skeleton + hands</b> | <b>Generated</b> | <b>Real video (never seen)</b></p>'
                     f'{video_tag(n)}{strip}</div>')
        b.append('</div>')
        blocks.append(''.join(b))
    if blocks:
        legend = ('<p class="legend small"><span><i class="sw" style="background:#ffe63c"></i>lips</span>'
                  '<span><i class="sw" style="background:#ffb454"></i>left hand</span>'
                  '<span><i class="sw" style="background:#6cb6ff"></i>right hand</span>'
                  '<span><i class="sw" style="background:#fff"></i>body skeleton</span>'
                  '<span><i class="sw" style="background:#a0968c"></i>face mesh</span></p>')
        clipsec = ('<p class="mut small">Each clip: what the model was given (left), what it made (middle), the real recording it never saw (right). '
                   'Tap play, sound on. Go full screen and rotate the phone for detail. Hands are drawn only when the tracker found them.</p>' + legend + ''.join(blocks))
    else:
        clipsec = '<p class="mut">No clips yet. Side-by-side videos appear here once a run has produced output for the held-out clips.</p>'

    summ = md(tload(os.path.join(pagedir, 'summary.md')))
    summary = f'<div class="card sum">{summ}</div>' if summ else '<div class="card sum"><p class="mut">Summary not written yet.</p></div>'
    nruns = len([r for r in rows.values() if not r['bust'] and not r['ref']])
    dummy_banner = '<div class="card" style="border-color:#ffb454"><b>DUMMY DATA</b>: layout test only, numbers and videos are fake.</div>' if args.dummy else ''
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>pose2real torso: landmarks, skeleton, hands and audio to talking video</title>
<style>{CSS}</style><script>{JS}</script></head><body><main>
<div id="err"></div>
<h1>pose2real, torso framing: landmarks, skeleton, hands and audio to a talking video</h1>
<p class="mut small">Held-out Seamless clips, 8 s, 512&times;640, 30 fps, head to hips. {nruns} run{'s' if nruns != 1 else ''} scored. Built {time.strftime('%Y-%m-%d %H:%M')}.</p>
{dummy_banner}{summary}
<section id="headline"><h2>Headline</h2>{headline(rows)}</section>
<section id="clips"><h2>Side by side, with sound</h2>{clipsec}</section>
<section id="ablation"><h2>All runs</h2>{ablation(rows)}{how_to_read(root)}</section>
<section id="bust"><h2>Torso vs bust</h2>{bust_html}</section>
{text_section('What worked', 'worked', tload(os.path.join(pagedir, 'worked.md')))}
{text_section('What did not work', 'didnt', tload(os.path.join(pagedir, 'didnt.md')))}
{text_section('Resolution vs speed', 'speed', tload(os.path.join(pagedir, 'speed.md')))}
{text_section('Next', 'next', tload(os.path.join(pagedir, 'next.md')))}
<section id="spend"><h2>Spend</h2><div class="card">{spend_section(root)}</div></section>
</main></body></html>
"""
    with open(os.path.join(out, 'index.html'), 'w') as f: f.write(page)
    ref = set(re.findall(r'videos/([^"]+)', page))
    for fn in os.listdir(vdir):
        base = re.sub(r'\.ver(\.cap)?$', '', fn)
        if base not in ref:
            try: os.remove(os.path.join(vdir, fn))
            except OSError: pass
    missing = [r for r in ref if not os.path.exists(os.path.join(vdir, r))]
    print(f'wrote {out}/index.html  runs={nruns} showcase_videos={nvid} jobs={len(jobs)} missing_files={missing}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dummy', action='store_true')
    ap.add_argument('--out')
    ap.add_argument('--no-videos', action='store_true')
    ap.add_argument('--force', action='store_true', help='re-encode all videos')
    build(ap.parse_args())
