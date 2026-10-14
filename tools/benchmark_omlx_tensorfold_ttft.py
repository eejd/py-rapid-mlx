#!/usr/bin/env python3
"""Same-host Qwen27 TTFT comparison; see the 2026-10-14 performance report.

Requires prepared app bundles, pinned TensorFold extensions and existing HF
snapshots. Never downloads models. Uses synthetic prompts and a loopback API.
"""
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import time
import urllib.request

ROOT=Path(os.environ.get('RAPID_TTFT_BENCH_ROOT', '/private/tmp/rapid-mlx-omlx-ttft'))
TARGET=Path.home()/'.cache/huggingface/hub/models--Vontra--Qwen3.8-27B-MLX-4bit'
SHA='70ae7fac63274ff2eac54152031433374cb80f2f'
assert (TARGET/'snapshots'/SHA/'model.safetensors.index.json').exists()
(TARGET/'refs').mkdir(exist_ok=True)
(TARGET/'refs/main').write_text(SHA)
models=ROOT/'model-links'
models.mkdir(exist_ok=True)
link=models/'qwen27'
if not link.exists():link.symlink_to(TARGET/'snapshots'/SHA)
key=secrets.token_hex(24)
rows=[]

def messages(wave,reference=False):
    first='QuickSilver TF acceptance wave %s. Reply with only the number 4.'%wave
    if not reference:return [{'role':'user','content':first}]
    return [{'role':'user','content':first},{'role':'assistant','content':'4'},
            {'role':'user','content':'Reference text: '+('alpha beta gamma delta. '*140)+' Reply with only the number 4.'}]

def call(config,name,content,record=True):
    payload=dict(model='qwen27',messages=content,temperature=0,max_tokens=32,stream=True,
                 stream_options={'include_usage':True},chat_template_kwargs={'enable_thinking':False})
    start=time.monotonic()
    first=None
    usage=None
    done=False
    text=''
    finish=None
    req=urllib.request.Request('http://127.0.0.1:18761/v1/chat/completions',data=json.dumps(payload).encode(),
        headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data:'):continue
            value=line[5:].strip()
            if value==b'[DONE]':done=True;break
            item=json.loads(value)
            assert not item.get('error'),item.get('error')
            usage=item.get('usage') or usage
            for choice in item.get('choices',[]):
                delta=choice.get('delta',{}).get('content') or ''
                if delta and first is None:first=time.monotonic()-start
                text+=delta
                finish=choice.get('finish_reason') or finish
    row=dict(config=config,name=name,ttft=round(first,4) if first else None,elapsed=round(time.monotonic()-start,4),
             usage=usage,done=done,answer_ok=text.strip()=='4',finish_reason=finish)
    assert done and first is not None and usage and row['answer_ok'], row
    if record:
        rows.append(row)
        (ROOT/'comparison-results.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(row),flush=True)
    return text

for config in ['omlx-no-cache','omlx-tiered-cache','rapid-tensorfold']:
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1')
    if config.startswith('omlx'):
        resources=ROOT/'oMLX.app/Contents/Resources'
        env['PYTHONPATH']=str(resources)+':'+str(resources/'Python/framework-mlx-base/lib/python3.11/site-packages')
        env.pop('PYTHONHOME',None)
        base=ROOT/config
        base.mkdir(exist_ok=True)
        (base/'model_settings.json').write_text(json.dumps({'version':1,'models':{'qwen27':{'model_type_override':'llm','enable_thinking':False}}}))
        args=[str(resources/'Python/cpython-3.11/bin/python3'),'-m','omlx.cli','serve',
              '--model-dir',str(models),'--no-hf-cache','--base-path',str(base),'--headless',
              '--host','127.0.0.1','--port','18761','--max-concurrent-requests','1']
        if config=='omlx-no-cache':args+=['--no-cache']
        else:args+=['--paged-ssd-cache-dir',str(base/'ssd-cache'),'--paged-ssd-cache-max-size','4GB']
    else:
        resources=ROOT/'Rapid-MLX Desktop.app/Contents/Resources/rapid-mlx'
        env['PYTHONHOME']=str(resources/'python')
        env['PYTHONPATH']=str(ROOT/'tf-extensions')+':'+str(resources/'site-packages')
        env['RAPID_MLX_API_KEY']=key
        args=[str(resources/'python/bin/python3.12'),'-P','-u','-s','-m','rapid_mlx.cli','serve','qwen3.8-27b-tensorfold',
              '--host','127.0.0.1','--port','18761','--max-num-seqs','1','--disable-disk-caches','--no-thinking',
              '--served-model-name','qwen27']
    with (ROOT/(config+'.log')).open('wb') as log:
        p=subprocess.Popen(args,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'active-server.json').write_text(json.dumps({'pid':p.pid,'config':config}))
        try:
            for _ in range(180):
                assert p.poll() is None,'server exited: '+config
                try:
                    req=urllib.request.Request('http://127.0.0.1:18761/v1/models',headers={'Authorization':'Bearer '+key})
                    with urllib.request.urlopen(req,timeout=2) as response:info=json.load(response)
                    assert any(m['id']=='qwen27' for m in info['data']),info
                    break
                except Exception:time.sleep(2)
            else:raise RuntimeError('server readiness timeout')
            call(config,'warmup-short',messages('warmup'),False)
            call(config,'warmup-prefill',messages('warmup-prefill',True),False)
            for wave in range(3):
                call(config,'short-%d'%wave,messages(wave))
                original=messages(wave,True)
                answer=call(config,'prefill-cold-%d'%wave,original)
                call(config,'prefill-repeat-%d'%wave,original)
                continued=original+[{'role':'assistant','content':answer},{'role':'user','content':'Reply with only the number 4 again.'}]
                call(config,'continuation-%d'%wave,continued)
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid,signal.SIGKILL)
                    p.wait()
            time.sleep(3)
