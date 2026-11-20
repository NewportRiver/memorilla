import os,subprocess,sys
from pathlib import Path
from run_lite007l import CACHE,CHILD,build_order
ROOT=Path(__file__).resolve().parents[2]; order,_=build_order(); os.environ.update({'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}); prev=CACHE/'direct-initial'/'checkpoint'
for ep in range(30):
 r=subprocess.run([sys.executable,str(CHILD),'child','direct',str(ep),str(prev)],cwd=ROOT,env=os.environ.copy(),check=False)
 if r.returncode: raise SystemExit(f'episode {ep} failed {r.returncode}')
 prev=CACHE/'direct'/f'episode-{ep:03d}'/'checkpoint'
print('direct complete',prev)
