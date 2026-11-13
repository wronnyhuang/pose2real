"""Contact sheet straight from the source video + crop (no cut needed). usage: sheet_raw.py crops.json out.jpg ids... (or 'auto N')"""
import json, sys, cv2, numpy as np
cr = json.load(open(sys.argv[1])); out = sys.argv[2]; ids = sys.argv[3:]
def tile(v, fr, extra=''):
    c = v['crop']; cap = cv2.VideoCapture(v['src_video']); cap.set(cv2.CAP_PROP_POS_FRAMES, v['st'] + fr); ok, f = cap.read()
    t = cv2.resize(f[c['y']:c['y'] + c['h'], c['x']:c['x'] + c['w']], (160, 200), interpolation=cv2.INTER_AREA)
    return t
tiles = []
for i in ids:
    v = cr[i]; a = tile(v, 60); b = tile(v, 180); t = np.hstack([a, b])
    cv2.putText(t, f"{i[:11]} s{v['crop']['scale']:.2f}", (3, 13), 0, 0.42, (0, 255, 255), 1); tiles.append(t)
n = 6; tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % n)
cv2.imwrite(out, np.vstack([np.hstack(tiles[i:i + n]) for i in range(0, len(tiles), n)]), [cv2.IMWRITE_JPEG_QUALITY, 82])
