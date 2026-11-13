#!/usr/bin/env python3
"""RunPod helper for torso_up. State in fast/pod_state.json. Never touches other pods."""
import json, os, sys, time, subprocess, pathlib, urllib.request

HERE = pathlib.Path(__file__).resolve().parent
STATE = HERE / ('state%s.json' % ('_' + os.environ['POD'] if os.environ.get('POD') else ''))
KEY = __import__('os').path.expandvars('$HOME/services/gpu-offload/id_ed25519')
NAME = 'p2r-torso-' + os.environ.get('POD', 'a')
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
GPUS = os.environ.get('GPUS', 'NVIDIA GeForce RTX 5090,NVIDIA GeForce RTX 4090,NVIDIA RTX PRO 4500 Blackwell,NVIDIA GeForce RTX 3090').split(',')
CAP = float(os.environ.get('CAP', '0.95'))


def gql(q, v=None):
    r = urllib.request.Request('https://api.runpod.io/graphql', data=json.dumps({'query': q, 'variables': v or {}}).encode(),
                               headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'], 'content-type': 'application/json'})
    with urllib.request.urlopen(r, timeout=40) as f:
        d = json.load(f)
    if d.get('errors'):
        raise RuntimeError(json.dumps(d['errors'])[:500])
    return d['data']


def st():
    try: return json.loads(STATE.read_text())
    except FileNotFoundError: return {}


def save(s): STATE.write_text(json.dumps(s, indent=1))


def balance():
    return gql('{ myself { clientBalance currentSpendPerHr } }')['myself']


def create():
    s = st()
    if s.get('pod_id') and not s.get('terminated_at'):
        print('already have pod', s['pod_id']); return
    b = balance(); print('balance', b)
    if b['clientBalance'] < 4.5: sys.exit('balance too low')
    last = None
    for g in GPUS:
        for cloud in ('COMMUNITY', 'SECURE'):
            try:
                d = gql('''mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) { id costPerHr machine { gpuDisplayName } } }''',
                        {'i': {'cloudType': cloud, 'gpuCount': 1, 'volumeInGb': 0, 'containerDiskInGb': 60, 'minVcpuCount': 8,
                               'minMemoryInGb': 30, 'gpuTypeId': g, 'name': NAME, 'imageName': IMAGE, 'ports': '22/tcp',
                               'startSsh': True, 'supportPublicIp': True, 'env': [{'key': 'NVIDIA_DRIVER_CAPABILITIES', 'value': 'all'}]}})
                p = d['podFindAndDeployOnDemand']
                if p is None: continue
                if p['costPerHr'] > CAP:
                    gql('mutation { podTerminate(input:{podId:"%s"}) }' % p['id']); print('too pricey', g, cloud, p['costPerHr']); continue
                s.update(pod_id=p['id'], rate=p['costPerHr'], gpu=g, cloud=cloud, created_at=time.time(), terminated_at=None)
                save(s); print('created', p); return
            except Exception as e:
                last = e; print('fail', g, cloud, str(e)[:200])
    sys.exit('no pod: %s' % last)


def info():
    s = st()
    d = gql('query($i: PodFilter) { pod(input: $i) { id name desiredStatus costPerHr runtime { uptimeInSeconds ports { ip isIpPublic privatePort publicPort } gpus { id } } } }',
            {'i': {'podId': s['pod_id']}})
    return d['pod']


def ssh_target():
    p = info()
    for x in (p.get('runtime') or {}).get('ports') or []:
        if x['privatePort'] == 22 and x['isIpPublic']:
            return x['ip'], x['publicPort']
    return None


def wait_ssh():
    for _ in range(90):
        t = ssh_target()
        if t:
            r = subprocess.run(ssh_cmd(t) + ['true'], capture_output=True)
            if r.returncode == 0:
                s = st(); s['ssh'] = list(t); save(s); print('ssh ok', t); return t
        time.sleep(8)
    sys.exit('ssh timeout')


def ssh_cmd(t=None):
    t = t or st()['ssh']
    return ['ssh', '-i', KEY, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
            '-o', 'LogLevel=ERROR', '-o', 'ServerAliveInterval=30', '-o', 'ConnectTimeout=15', '-p', str(t[1]), 'root@' + t[0]]


def spend():
    s = st()
    end = s.get('terminated_at') or time.time()
    h = (end - s['created_at']) / 3600
    return {'usd': round(h * s['rate'], 3), 'pod_hours': round(h, 3), 'rate': s['rate'], 'gpu': s.get('gpu'), 'cloud': s.get('cloud')}


def terminate():
    s = st()
    gql('mutation { podTerminate(input:{podId:"%s"}) }' % s['pod_id'])
    time.sleep(5)
    pods = gql('{ myself { pods { id name } } }')['myself']['pods']
    gone = all(p['id'] != s['pod_id'] for p in pods)
    if gone:
        s['terminated_at'] = time.time(); save(s)
    print('terminated' if gone else 'STILL PRESENT', [p['name'] for p in pods])


if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'create': create()
    elif c == 'wait': wait_ssh()
    elif c == 'info': print(json.dumps(info(), indent=1))
    elif c == 'balance': print(balance())
    elif c == 'spend': print(json.dumps(spend()))
    elif c == 'ssh': os.execvp('ssh', ssh_cmd() + sys.argv[2:])
    elif c == 'sshargs': print(' '.join(ssh_cmd()[:-1]) + '|' + ssh_cmd()[-1])
    elif c == 'terminate': terminate()
