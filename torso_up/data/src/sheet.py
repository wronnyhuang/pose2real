"""Contact sheets from clips/<id>/real.mp4 (+ lm.npz overlay: hands green=left / red=right, pose wrists blue).
usage: sheet.py heldout | refs | train | all | ids <out.jpg> <id>...
  heldout -> contact_heldout.jpg (20 clips x frames 30/120/210), refs -> contact_refs.jpg, train -> contact_train.jpg
  (per identity up to 4 clips, frame 120), all -> contact_all.jpg (every manifest clip, 80x100 tile, frame 120, no overlay)"""
import json, sys, os, glob, collections, cv2, numpy as np
D = __import__('os').path.expandvars('$WORK/data')


def frame(cid, i, w=192, overlay=True):
    cap = cv2.VideoCapture(f'{D}/clips/{cid}/real.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    if not ok: return np.zeros((w * 5 // 4, w, 3), np.uint8)
    if overlay and os.path.exists(f'{D}/clips/{cid}/lm.npz'):
        d = np.load(f'{D}/clips/{cid}/lm.npz')
        for s, col in ((0, (0, 255, 0)), (1, (0, 0, 255))):
            if d['hands_ok'][i, s]:
                for x, y in d['hands'][i, s, :, :2] * [512, 640]: cv2.circle(f, (int(x), int(y)), 3, col, -1)
        if d['pose_ok'][i]:
            for k in (15, 16):
                x, y = d['pose'][i, k, :2] * [512, 640]; cv2.circle(f, (int(x), int(y)), 6, (255, 0, 0), 2)
    return cv2.resize(f, (w, w * 5 // 4), interpolation=cv2.INTER_AREA)


def meta(cid):
    p = f'{D}/clips/{cid}/meta.json'
    return json.load(open(p)) if os.path.exists(p) else {}


def label(t, cid, extra=''):
    cv2.putText(t, cid[:11], (3, 13), 0, 0.42, (0, 255, 255), 1)
    if extra: cv2.putText(t, extra, (3, t.shape[0] - 5), 0, 0.38, (0, 255, 255), 1)
    return t


def grid(tiles, n):
    tiles = tiles + [np.zeros_like(tiles[0])] * (-len(tiles) % n)
    return np.vstack([np.hstack(tiles[i:i + n]) for i in range(0, len(tiles), n)])


if __name__ == '__main__':
    mode = sys.argv[1]
    plan = json.load(open(f'{D}/work/plan.json'))
    if mode in ('heldout', 'refs'):
        ids = [w['id'] for w in plan if w['split'] == ('heldout' if mode == 'heldout' else 'heldout_ref') and os.path.exists(f"{D}/clips/{w['id']}/qc.json")]
        tiles = []
        for cid in ids:
            m = meta(cid)
            t = np.hstack([frame(cid, i, 160) for i in (30, 120, 210)])
            tiles.append(label(t, cid, f"s{m['crop']['scale']:.2f} hand {m['hand_frac']:.2f}/{m['hand2_frac']:.2f} face {m['face_width_px']:.0f}px"))
        cv2.imwrite(f'{D}/contact_{mode}.jpg', grid(tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 85])
    elif mode == 'train':
        man = json.load(open(f'{D}/manifest.json'))
        by = collections.defaultdict(list)
        for c in man['clips']:
            if c['split'] == 'train': by[c['identity']].append(c)
        tiles = []
        for pid in sorted(by):
            cs = by[pid]; g = [c for c in cs if c['set'] == 'gesture']; pr = [c for c in cs if c['set'] != 'gesture']
            pick = (g[:2] + pr[::max(1, len(pr) // 3)])[:4]
            for c in pick:
                m = meta(c['id']); tiles.append(label(frame(c['id'], 120, 128), c['id'], f"{c['set'][0]} h{m['hand_frac']:.2f}"))
        n = 16; rows = grid(tiles, n); h = rows.shape[0]; half = (h // 160 + 1) // 2 * 160
        cv2.imwrite(f'{D}/contact_train_a.jpg', rows[:half], [cv2.IMWRITE_JPEG_QUALITY, 82])
        cv2.imwrite(f'{D}/contact_train_b.jpg', rows[half:], [cv2.IMWRITE_JPEG_QUALITY, 82])
    elif mode == 'all':
        man = json.load(open(f'{D}/manifest.json'))
        tiles = [frame(c['id'], 120, 64, False) for c in man['clips']]
        cv2.imwrite(f'{D}/contact_all.jpg', grid(tiles, 40), [cv2.IMWRITE_JPEG_QUALITY, 78])
    elif mode == 'ids':
        out = sys.argv[2]; tiles = []
        for cid in sys.argv[3:]:
            m = meta(cid); tiles.append(label(np.hstack([frame(cid, i, 128) for i in (40, 200)]), cid, f"h{m.get('hand_frac', 0):.2f} {m.get('qc', {}).get('drop', '')[:20]}"))
        cv2.imwrite(out, grid(tiles, 6), [cv2.IMWRITE_JPEG_QUALITY, 82])
