"""Contact sheets: contact_sheet.jpg (per identity row: up to 8 clips, mid frame, split-coloured label; heldout+refs first)
and contact_all.jpg (every manifest clip, 96 px tile, frame 120)."""
import json, collections, cv2, numpy as np
D = __import__('os').path.expandvars('$WORK/data')
man = json.load(open(f'{D}/manifest.json'))
def frame(cid, i=120, s=128):
    cap = cv2.VideoCapture(f'{D}/clips/{cid}/real.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, i); ok, f = cap.read()
    return cv2.resize(f, (s, s), interpolation=cv2.INTER_AREA) if ok else np.zeros((s, s, 3), np.uint8)
col = {'heldout': (0, 0, 255), 'heldout_ref': (0, 165, 255), 'train': (0, 200, 0)}
byid = collections.defaultdict(list)
for c in man['clips']: byid[c['identity']].append(c)
ho = man['heldout_identities']; ids = ho + sorted(k for k in byid if k not in ho)
rows = []
for pid in ids:
    cs = byid[pid]; pick = [c for c in cs if c['split'] != 'train'][:8]
    tr = [c for c in cs if c['split'] == 'train']; pick += tr[::max(1, len(tr) // max(1, 8 - len(pick)))][:8 - len(pick)]
    tiles = []
    for c in pick:
        t = frame(c['id']); cv2.rectangle(t, (0, 0), (127, 127), col[c['split']], 3)
        cv2.putText(t, c['id'][:5], (4, 14), 0, 0.4, (255, 255, 255), 1); tiles.append(t)
    lab = np.full((128, 110, 3), 30, np.uint8); cv2.putText(lab, pid, (4, 50), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(lab, f'n={len(cs)}', (4, 80), 0, 0.45, (200, 200, 200), 1)
    tiles += [np.zeros((128, 128, 3), np.uint8)] * (8 - len(tiles)); rows.append(np.hstack([lab] + tiles))
half = (len(rows) + 1) // 2; L = rows[:half]; R = rows[half:] + [np.zeros_like(rows[0])] * (half - len(rows[half:]))
cv2.imwrite(f'{D}/contact_sheet.jpg', np.hstack([np.vstack(L), np.full((128 * half, 8, 3), 255, np.uint8), np.vstack(R)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
tiles = [frame(c['id'], 120, 96) for c in man['clips']]; n = 32
tiles += [np.zeros((96, 96, 3), np.uint8)] * (-len(tiles) % n)
cv2.imwrite(f'{D}/contact_all.jpg', np.vstack([np.hstack(tiles[i:i + n]) for i in range(0, len(tiles), n)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print('identities', len(ids), 'clips', len(man['clips']))
