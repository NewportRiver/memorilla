from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import load_model
from lite.training.numpy_gradients import score_head_forward
from lite.training.direct_bilinear import load as load_direct
from lite.training.resumable_evaluation import ResumableEvaluation

def main():
 model_name,ckpt,milestone=sys.argv[1],sys.argv[2],sys.argv[3]; base=Path.home()/'.cache/memorilla-lite/LITE-007J'; data=json.loads(Path('benchmarks/lite/fixtures/lite007i/dataset.json').read_text()); a=np.load(base.parent/'LITE-007I/embeddings.npz',allow_pickle=False); rid={x:i for i,x in enumerate(a['record_ids'].tolist())}; qid={x:i for i,x in enumerate(a['query_ids'].tolist())}; R=a['record_embeddings']; Q=a['query_embeddings']; sc={x['scenario_id']:x for x in data['scenarios']}; queries=[q for q in data['queries'] if sc[q['scenario_id']]['split']=='validation']; records=[x for x in data['records']]; model=load_model(ckpt)[0] if model_name=='associative' else None; W=load_direct(ckpt)[0] if model_name=='direct' else None; ids=[q['query_id'] for q in queries]; root=base/f'validation-{model_name}-{milestone}'; ev=ResumableEvaluation(root,f'J-{model_name}-{milestone}',f'{model_name}:{ckpt}',data['dataset_version'],'validation',ids,batch_size=16)
 def batch(xs):
  out=[]
  for qidv in xs:
   q=next(x for x in queries if x['query_id']==qidv); s=sc[q['scenario_id']]; es=s['record_ids']; M=R[[rid[e] for e in es]]; qv=Q[qid[qidv]]; req={es.index(e) for e in q['required_evidence_ids'] if e in es}; scores=(score_head_forward(model.forward(qv,M,return_latent=True)[1],M,model.state,np.ones(len(es),bool))[0] if model_name=='associative' else qv@W@M.T); order=np.argsort(-scores); ranks={int(x):i+1 for i,x in enumerate(order)}; out.append({'collection_size':s['collection_size'],'category':q['category'],'required_count':len(req),'recall_at_1':float(bool(set(order[:1])&req)),'recall_at_3':float(bool(set(order[:3])&req)),'recall_at_5':float(bool(set(order[:5])&req)),'mrr':min((1/ranks[x] for x in req if x in ranks),default=0.0),'complete_required_top5':float(bool(req) and req.issubset(set(order[:5])))})
  return out
 r=ev.run(batch); Path(root/'aggregate.json').write_text(json.dumps(r,sort_keys=True,indent=2)+'\n'); print(model_name,milestone,r['denominator'])
if __name__=='__main__': main()
