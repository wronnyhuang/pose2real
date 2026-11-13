# start_pod.sh <audio list> : subset manifest of complete clips -> prep -> queue (idempotent)
[ -f /root/STARTED ] && { echo already started; exit 0; }
touch /root/STARTED
cd /root/code && python - <<'P'
import json,os,collections
m=json.load(open('/root/data/manifest.json'))
ok=lambda c,f: os.path.exists('/root/data/clips/%s/%s'%(c['id'],f)) and os.path.getsize('/root/data/clips/%s/%s'%(c['id'],f))>1000
have=[c for c in m['clips'] if ok(c,'lm.npz') and ok(c,'real.mp4') and ok(c,'audio_pad.wav')]
print(len(have), collections.Counter(c['split'] for c in have), collections.Counter(c['set'] for c in have))
os.makedirs('/root/dsub',exist_ok=True)
m['clips']=have; json.dump(m,open('/root/dsub/manifest.json','w'))
os.path.exists('/root/dsub/clips') or os.symlink('/root/data/clips','/root/dsub/clips')
P
export DATA=/root/dsub FRAMES=/root/frames
setsid nohup bash -c "python prep.py --audio $1 --dev cuda > /root/prep.log 2>&1; /root/code/q.sh /root/queue.txt > /root/q.out 2>&1" > /dev/null 2>&1 < /dev/null &
echo launched
