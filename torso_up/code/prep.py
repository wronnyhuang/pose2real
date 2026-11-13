"""Pod-side preparation in one command (resumable: finished clips are skipped, feature files are written atomically).
    DATA=/root/data FRAMES=/root/frames python prep.py [--audio loc,ctx,win] [--splits heldout,heldout_ref,train] [--dev cuda]
frames -> $FRAMES/<id>/%04d.jpg (+ ok marker); HuBERT au_loc.npy / au_ctx.npy (50 Hz) and the causal windowed au_win.npy."""
import argparse, time
import data as D

ap = argparse.ArgumentParser()
ap.add_argument('--audio', default='loc,ctx,win')
ap.add_argument('--splits', default='heldout,heldout_ref,train')   # heldout first: the test set is ready earliest
ap.add_argument('--dev', default='cuda')
ap.add_argument('--jobs', type=int, default=16)      # parallel ffmpeg
ap.add_argument('--max_clips', type=int, default=0)
a = ap.parse_args()
sp = a.splits.split(',')
ids = [c['id'] for s in sp for c in D.manifest() if c['split'] == s]
if a.max_clips: ids = ids[:a.max_clips]
t0 = time.time()
D.prep_frames(ids, a.jobs); print(f'frames ok ({len(ids)} clips) {time.time() - t0:.0f}s', flush=True)
au = a.audio.split(',')
if 'loc' in au or 'ctx' in au: D.prep_audio(ids, a.dev); print(f'audio loc/ctx ok {time.time() - t0:.0f}s', flush=True)
if 'win' in au: D.prep_audio_win(ids, a.dev); print(f'audio win ok {time.time() - t0:.0f}s', flush=True)
for x in au: print(x, 'ready', len(D.ready_ids(sp, x)), 'of', len(ids), flush=True)
print('PREP_OK', flush=True)
