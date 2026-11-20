from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.sample_scheduler import PersistentSampleScheduler

def main():
 root=Path(tempfile.mkdtemp(prefix='lite007k-scheduler-')); prepared=Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-train.npz'; ds=PreparedTrainingSet(prepared); eligible=ds.eligible_indices(); by={8:[],16:[],32:[]}; data=json.loads(Path('benchmarks/lite/fixtures/lite007i/dataset.json').read_text()); sc={x['scenario_id']:x for x in data['scenarios']}
 for i in eligible: by[sc[ds.get(i)['scenario_id']]['collection_size']].append(i)
 for v in by.values(): v.sort()
 selected=[by[8][i] for i in range(80)]+[by[16][i] for i in range(80)]+[by[32][i] for i in range(80)]
 order=selected+[i for i in eligible if i not in set(selected)]
 order_path=root/'order.json'; order_path.write_text(json.dumps(order)); eligible_path=root/'eligible.json'; eligible_path.write_text(json.dumps(eligible))
 script=Path('benchmarks/lite/_scheduler_child.py').resolve()
 script.write_text("import json,sys\nfrom lite.training.sample_scheduler import PersistentSampleScheduler\np= sys.argv[1]; ident=sys.argv[2]; order=json.loads(open(sys.argv[3]).read()); eligible=json.loads(open(sys.argv[4]).read()); s=PersistentSampleScheduler(p,ident,eligible,order=order); s.reserve(8,sys.argv[5]); sys.exit(97 if sys.argv[6]=='interrupt' else 0)\n")
 results={}
 for model in ['associative','direct']:
  path=root/f'{model}.json'; ident='prepared-train-identity'; exits=[]
  for ep in range(30):
   mode='interrupt' if ep==10 else 'complete'; r=subprocess.run([sys.executable,str(script),str(path),ident,str(order_path),str(eligible_path),f'{model}-episode-{ep:02d}',mode]); exits.append(r.returncode)
   if mode=='interrupt' and r.returncode!=97: raise RuntimeError('fault injection boundary failed')
   if mode=='complete' and r.returncode!=0: raise RuntimeError('subprocess reservation failed')
  state=json.loads(path.read_text()); reserved=[i for x in state['reservations'] for i in x['indices']]; counts={str(k):0 for k in by}
  for i in reserved[:240]: counts[str(next(k for k,v in by.items() if i in v))]+=1
  results[model]={'episodes':30,'reserved_updates':state['cursor'],'unique_reserved_first_240':len(set(reserved[:240])),'collection_size_counts_first_240':counts,'order_digest':state['order_digest'],'dataset_identity':state['dataset_identity'],'subprocess_exit_codes':exits,'interruption_recovered':exits[10]==97 and len(state['reservations'])==30}
 ds.close(); out=Path('docs/memorilla-lite/audits/LITE-007K-SCHEDULER-QUALIFICATION.json'); out.write_text(json.dumps({'eligible_inventory':{str(k):len(v) for k,v in by.items()},'eligible_total':len(eligible),'target_unique_examples':240,'target_per_collection_size':80,'parameter_mutations':0,'models':results},sort_keys=True,indent=2)+'\n'); print(json.dumps(results,indent=2))
if __name__=='__main__': main()
