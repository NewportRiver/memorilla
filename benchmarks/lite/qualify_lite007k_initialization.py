import json
from pathlib import Path
import numpy as np
from lite.training.evaluation_metrics import ranking_metrics
from lite.training.direct_bilinear import load

def main():
 d=json.loads(Path('benchmarks/lite/fixtures/lite007i/dataset.json').read_text()); a=np.load(Path.home()/'.cache/memorilla-lite/LITE-007I/embeddings.npz',allow_pickle=False); rid={x:i for i,x in enumerate(a['record_ids'].tolist())}; qid={x:i for i,x in enumerate(a['query_ids'].tolist())}; R=a['record_embeddings']; Q=a['query_embeddings']; sc={x['scenario_id']:x for x in d['scenarios']}; qs=[q for q in d['queries'] if sc[q['scenario_id']]['split']=='validation']; zero,_=load(Path.home()/'.cache/memorilla-lite/LITE-007J/direct/initial'); eye=np.eye(384,dtype=np.float32)
 def eval_kind(kind,permute=False):
  grouped={}
  for q in qs:
   s=sc[q['scenario_id']]; es=list(s['record_ids']); M=R[[rid[e] for e in es]]; qv=Q[qid[q['query_id']]]
   if permute: M=M[::-1]; es=es[::-1]
   if kind=='zero': scores=qv@zero@M.T
   elif kind=='identity': scores=qv@eye@M.T
   else: scores=(qv/np.maximum(np.linalg.norm(qv),1e-9))@(M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9)).T
   order=np.argsort(-scores,kind='mergesort'); req=[es.index(e) for e in q['required_evidence_ids'] if e in es]; row={'ranked_ids':order.tolist(),'required_ids':req}; grouped.setdefault(str(s['collection_size']),[]).append(row)
  return {k:ranking_metrics(v) for k,v in grouped.items()}
 out={k:eval_kind(k) for k in ['zero','identity','cosine']}; out['zero_permuted']=eval_kind('zero',True); out['provenance']={'historical_direct_initial_checkpoint':'zero matrix, sha256 250480ecefa3655c774b6a4e5d4eb16675ee4f181c2285c25020fac6e3aaf3b5','identity_variant':'read-only np.eye(384)','weights_mutated':False,'training_updates':0}; Path('docs/memorilla-lite/audits/LITE-007K-INITIALIZATION-QUALIFICATION.json').write_text(json.dumps(out,sort_keys=True,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
