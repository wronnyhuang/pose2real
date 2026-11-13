"""cut(video, st_frame, crop, out_mp4): exact 240 frames at 30 fps, static crop, lanczos to 512x640, crf 16 (same seek as PRIOR)."""
import subprocess, os
def cut(video, st, c, out, crf=16):
    vf = f"trim=start_frame=0:end_frame=240,setpts=PTS-STARTPTS,crop={c['w']}:{c['h']}:{c['x']}:{c['y']},scale=512:640:flags=lanczos"
    tmp = out + '.tmp.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '2', '-ss', f'{st/30:.4f}', '-i', video, '-vf', vf, '-an', '-c:v', 'libx264',
                    '-preset', 'fast', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-r', '30', '-y', tmp], check=True)
    os.replace(tmp, out)
def kp_window(video, st):
    import numpy as np
    base = os.path.dirname(os.path.dirname(video)); fid = os.path.basename(video)[:-4]
    kp = np.load(f'{base}/boxes_and_keypoints/keypoints/{fid}.npy', mmap_mode='r'); vb = np.load(f'{base}/boxes_and_keypoints/is_valid_box/{fid}.npy').reshape(-1)
    return np.array(kp[st:st + 240]), vb[st:st + 240]
