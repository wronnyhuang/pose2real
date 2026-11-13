cd /root/code && python - <<'P'
import json,os,collections
m=json.load(open('/root/data/manifest.json'))
ok=lambda c,f: os.path.exists('/root/data/clips/%s/%s'%(c['id'],f))
have=[c for c in m['clips'] if ok(c,'lm.npz') and ok(c,'real.mp4') and ok(c,'audio_pad.wav')]
print(len(have), collections.Counter(c['split'] for c in have))
os.makedirs('/root/dsmoke',exist_ok=True)
m['clips']=have; json.dump(m,open('/root/dsmoke/manifest.json','w'))
os.path.exists('/root/dsmoke/clips') or os.symlink('/root/data/clips','/root/dsmoke/clips')
P
export DATA=/root/dsmoke FRAMES=/root/fsmoke
python prep.py --audio loc 2>&1 | tail -4
