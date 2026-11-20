from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import load_model
from lite.training.numpy_gradients import score_head_forward
from lite.training.resumable_evaluation import ResumableEvaluation

def run(model_name, checkpoint=None):
 root=Path('benchmarks/lite/fixtures/lite007i'); cache=Path.home()/'.cache/memorilla-lite/LITE-007I'; data=json.loads((root/'dataset.json').read_text()); a=np.load(cache/'embeddings.npz',allow_pickle=False); rids=a['record_ids']; qids=a['query_ids']; re=a['record_embeddings']; qe=a['query_embeddings']; rid={x:i for i,x in enumerate(rids.tolist())}; qid={x:i for i,x in enumerate(qids.tolist())}; records={x['evidence_id']:x for x in data['records']}; scenarios={x['scenario_id']:x for x in data['scenarios']}; model=load_model(checkpoint)[0] if checkpoint else NumpyAssociativeMemory(NeuralConfig(seed=20261120)); queries=data['queries']; ids=[q['query_id'] for q in queries]; ev=ResumableEvaluation(cache/f'eval-{model_name}',model_name,model_name,'lite007i-dev-v1','all',ids,batch_size=8)
 def batch(batch_ids):
  out=[]
  for qidv in batch_ids:
   q=next(x for x in queries if x['query_id']==qidv); sc=scenarios[q['scenario_id']]; es=sc['record_ids']; mem=re[[rid[e] for e in es]]; qv=qe[qid[qidv]]; _,latent=model.forward(qv,mem,return_latent=True); scores=score_head_forward(latent,mem,model.state,np.ones(len(es),bool))[0]; order=np.argsort(-scores); req={es.index(x) for x in q['required_evidence_ids'] if x in es}; ranks={x:i+1 for i,x in enumerate(order)}; rr=min((1/ranks[x] for x in req if x in ranks),default=0.0); out.append({'collection_size':sc['collection_size'],'split':sc['split'],'category':q['category'],'required_count':len(req),'recall_at_1':float(bool(set(order[:1])&req)),'recall_at_3':float(bool(set(order[:3])&req)),'recall_at_5':float(bool(set(order[:5])&req)),'mrr':rr,'complete_required_top5':float(bool(req) and req.issubset(set(order[:5]))),'abstention_expected':bool(q['expected_abstention'])})
  return out
 result=ev.run(batch); (cache/f'eval-{model_name}-complete.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); print(model_name,result['denominator'])
if __name__=='__main__':
 name=sys.argv[1]; cp=sys.argv[2] if len(sys.argv)>2 and sys.argv[2]!='none' else None; run(name,cp)
