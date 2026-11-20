import json,re
from pathlib import Path
from collections import defaultdict
import numpy as np

def mean(rows):
 ks=['recall_at_1','recall_at_3','recall_at_5','mrr','complete_required_top5']; return {k:float(np.mean([x['value'][k] for x in rows])) for k in ks}|{'denominator':len(rows)}
def main():
 b=Path.home()/'.cache/memorilla-lite/LITE-007J'; curves=[]
 for model in ('associative','direct'):
  for n in (0,40,80,120,160,200,240):
   d=json.loads((b/f'validation-{model}-{n}/aggregate.json').read_text()); groups=defaultdict(list)
   for x in d['results']: groups[str(x['value']['collection_size'])].append(x)
   curves.append({'model':model,'updates':n,'metrics':{k:mean(v) for k,v in groups.items()},'denominator':d['denominator']})
 Path('docs/memorilla-lite/audits/LITE-007J-LEARNING-CURVES.json').write_text(json.dumps(curves,sort_keys=True,indent=2)+'\n')
 # representation summaries from checkpoints
 from lite.neural.serialization import load_model
 reps=[]
 for n in (0,40,80,120,160,200,240):
  p=b/'associative/initial' if n==0 else b/f'associative/episode-{n//8-1:03d}/checkpoint'; m,_=load_model(p); slots=[v for k,v in m.state.items() if 'slot' in k.lower() or 'memory' in k.lower()]; reps.append({'updates':n,'parameter_norm':float(np.sqrt(sum(np.sum(v*v) for v in m.state.values()))),'slot_norm':float(np.sqrt(sum(np.sum(v*v) for v in slots))) if slots else None,'slot_count':len(slots),'all_finite':all(np.isfinite(v).all() for v in m.state.values())})
 Path('docs/memorilla-lite/audits/LITE-007J-REPRESENTATION-ANALYSIS.json').write_text(json.dumps({'status':'task_level_and_weight_level_only','snapshots':reps,'attention_distribution':'not exposed by current native forward cache','slot_similarity':'not exposed by current checkpoint schema'},sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
