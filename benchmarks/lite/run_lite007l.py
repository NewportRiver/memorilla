from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.sample_scheduler import PersistentSampleScheduler

ROOT=Path(__file__).resolve().parents[2]; CACHE=Path.home()/'.cache/memorilla-lite/LITE-007L'; PREP=Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-train.npz'; SOURCE=ROOT/'benchmarks/lite/fixtures/lite007i/dataset.json'; CHILD=Path(__file__).resolve()

def build_order():
 ds=PreparedTrainingSet(PREP); eligible=ds.eligible_indices(); qids={i:ds.get(i)['query_id'] for i in eligible}; data=json.loads(SOURCE.read_text()); sizes={q['query_id']:str(next(s['collection_size'] for s in data['scenarios'] if s['scenario_id']==q['scenario_id'])) for q in data['queries']}; groups={x:sorted([i for i in eligible if sizes[qids[i]]==x]) for x in ('8','16','32')}; first=[]
 for n in range(80):
  for x in ('8','16','32'): first.append(groups[x][n])
 order=first+[i for i in eligible if i not in set(first)]; ds.close(); return order, eligible

def child(model, episode, initial, scheduler, order):
 os.environ.update({'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
 if model=='associative':
  from lite.training.operational_training import run_authorized_episode
  return run_authorized_episode(PREP,CACHE/model/f'episode-{episode:03d}',CACHE/f'{model}-budget.json','LITE-007L-associative',requested_updates=8,authorized=True,authorization_total_updates=240,initial_checkpoint=initial,scheduler_path=scheduler,schedule_order=order,min_ram=1280*1024**2,abort_ram=1024*1024**2,max_rss=256*1024**2,max_seconds=30)
 from lite.training.direct_bilinear import run_episode
 return run_episode(PREP,CACHE/model/f'episode-{episode:03d}',CACHE/'direct-budget-recovered.json','LITE-007L-direct',init=initial,requested=8,total=240,scheduler_path=scheduler,schedule_order=order)

def main():
 CACHE.mkdir(parents=True,exist_ok=True); order,eligible=build_order(); (CACHE/'schedule-order.json').write_text(json.dumps(order)); (CACHE/'eligible-indices.json').write_text(json.dumps(eligible));
 if len(sys.argv)>1 and sys.argv[1]=='child':
  model=sys.argv[2]; ep=int(sys.argv[3]); initial=Path(sys.argv[4]) if sys.argv[4]!='NONE' else None; return child(model,ep,initial,CACHE/f'{model}-scheduler.json',order)
 for model in ('associative','direct'):
  if model=='direct':
   from lite.training.direct_bilinear import create_initial_checkpoint
   create_initial_checkpoint(CACHE/'direct-initial'/'checkpoint',dimension=384,mode='zeros',seed=20261120)
  previous=None
  for ep in range(30):
   first=str(CACHE/'direct-initial'/'checkpoint') if model=='direct' and previous is None else (str(previous) if previous else 'NONE')
   r=subprocess.run([sys.executable,str(CHILD),'child',model,str(ep),first],cwd=ROOT,env=os.environ.copy(),check=False)
   if r.returncode!=0: raise SystemExit(f'{model} episode {ep} failed: {r.returncode}')
   previous=CACHE/model/f'episode-{ep:03d}'/'checkpoint'
  print(model,'complete',previous)
if __name__=='__main__': main()
