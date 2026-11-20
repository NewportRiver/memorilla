from __future__ import annotations
import json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

def summarize(rows):
 keys=['recall_at_1','recall_at_3','recall_at_5','mrr','complete_required_top5']
 return {k:float(np.mean([r[k] for r in rows])) for k in keys}|{'denominator':len(rows)}

def main():
 cache=Path.home()/'.cache/memorilla-lite/LITE-007I'; out={}
 files=['untrained','score_head_trained','memory_projection_trained','learned_slot_query','attention_ffn']
 for name in files:
  data=json.loads((cache/f'eval-{name}-complete.json').read_text()); groups=defaultdict(list)
  for x in data['results']: groups[(str(x['value']['collection_size']),x['value']['split'])].append(x['value'])
  out[name]={f'{s}:{p}':summarize(v) for (s,p),v in groups.items()}
 # Add deterministic untrained bilinear and query-independent controls.
 root=Path('benchmarks/lite/fixtures/lite007i'); d=json.loads((root/'dataset.json').read_text()); a=np.load(cache/'embeddings.npz',allow_pickle=False); rid={x:i for i,x in enumerate(a['record_ids'].tolist())}; qid={x:i for i,x in enumerate(a['query_ids'].tolist())}; R=a['record_embeddings']; Q=a['query_embeddings']; rec={x['evidence_id']:x for x in d['records']}; scs={x['scenario_id']:x for x in d['scenarios']}; rng=np.random.default_rng(20261120); W=rng.standard_normal((384,384),dtype=np.float32)/np.sqrt(384); groups=defaultdict(list); qi=defaultdict(list)
 for q in d['queries']:
  sc=scs[q['scenario_id']]; es=sc['record_ids']; mat=R[[rid[e] for e in es]]; qv=Q[qid[q['query_id']]]; req={es.index(x) for x in q['required_evidence_ids'] if x in es};
  for name,scores,target in [('direct_bilinear_untrained',qv@W@mat.T,groups),('query_independent',np.linalg.norm(mat,axis=1),qi)]:
   order=np.argsort(-scores); ranks={int(x):i+1 for i,x in enumerate(order)}; target[(str(sc['collection_size']),sc['split'])].append({'recall_at_1':float(bool(set(order[:1])&req)),'recall_at_3':float(bool(set(order[:3])&req)),'recall_at_5':float(bool(set(order[:5])&req)),'mrr':min((1/ranks[x] for x in req if x in ranks),default=0.0),'complete_required_top5':float(bool(req) and req.issubset(set(order[:5])) )})
 out['direct_bilinear_untrained']={f'{s}:{p}':summarize(v) for (s,p),v in groups.items()}; out['query_independent']={f'{s}:{p}':summarize(v) for (s,p),v in qi.items()}
 prior=json.loads(Path('docs/memorilla-lite/audits/LITE-007I-COMPARATIVE-EVALUATION.json').read_text())
 for name in ('lexical','direct_cosine','hybrid'):
  out[name]={f'{s}:combined':prior[str(s)][name] for s in (8,16,32,64,128)}
 (Path('docs/memorilla-lite/audits/LITE-007J-COMPARATIVE-EVALUATION.json')).write_text(json.dumps(out,sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
