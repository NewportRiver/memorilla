#!/data/data/com.termux/files/home/.cache/memorilla-lite/LITE-008D-R4/venv/bin/python
from __future__ import annotations
import argparse, hashlib, importlib, importlib.metadata as md, json, os, platform, shutil, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
VENVP=Path(__file__).resolve().parents[2].parent/'.cache/memorilla-lite/LITE-008D-R4/venv/bin/python'
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
TOKENIZER=MODEL/'tokenizer.json'
R2BUNDLE=Path.home()/'.cache/memorilla-lite/LITE-008V-R2/reconstruction.pt'

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--json',action='store_true'); a=ap.parse_args()
 names=['torch','numpy','onnxruntime','transformers','tokenizers','safetensors']
 required={'torch':'2.11.0','numpy':'2.4.4','onnxruntime':'1.27.0','transformers':'4.57.6','tokenizers':'0.22.2','safetensors':'0.8.0'}
 checks=[]; packages={}
 for n in names:
  try:
   packages[n]={'version':md.version(n),'location':str(Path(importlib.import_module(n).__file__).resolve())}; checks.append((n,packages[n]['version']==required[n]))
  except Exception as e: packages[n]={'error':repr(e)}; checks.append((n,False))
 files={}
 for p in [TOKENIZER,R2BUNDLE]: files[str(p)]={'exists':p.is_file(),'sha256':sha(p) if p.is_file() else None,'bytes':p.stat().st_size if p.is_file() else None}
 checks += [('interpreter_identity',Path(sys.prefix).resolve()==Path.home()/'.cache/memorilla-lite/LITE-008D-R4/venv'),('tokenizer',files[str(TOKENIZER)]['exists']),('r2_bundle',files[str(R2BUNDLE)]['exists']),('cpu_only',not bool(os.environ.get('CUDA_VISIBLE_DEVICES')) or os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1')),('one_thread_env',all(os.environ.get(k,'1') in ('1','-1') for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS'))),('storage',shutil.disk_usage(ROOT).free>2*1024**3),('memory',int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024>=536870912)]
 out={'schema':'LITE-008W-R1-preflight-v1','status':'PASS' if all(v for _,v in checks) else 'BLOCKED','interpreter':{'path':sys.executable,'version':sys.version,'platform':platform.platform(),'machine':platform.machine(),'venv_root':str(Path(sys.prefix))},'packages':packages,'files':files,'checks':dict(checks),'resource_classification':{'classification':'scientific_reference_evaluation','production_rss_ceiling_bytes':268435456,'scientific_rss_ceiling_bytes':1073741824,'training_rss_ceiling_bytes':1073741824,'minimum_available_bytes':536870912,'cpu_only':True,'numerical_threads':1,'subprocesses':1,'batch_size':1,'max_sequence_length':32,'latent_slots':1,'max_subprocess_seconds':300,'max_cumulative_seconds':1800,'optimizer_updates':0}}
 print(json.dumps(out,indent=2))
 raise SystemExit(0 if out['status']=='PASS' else 2)
if __name__=='__main__': main()
