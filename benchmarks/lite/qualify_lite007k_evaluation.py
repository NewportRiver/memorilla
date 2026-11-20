from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from lite.neural.serialization import load_model
from lite.training.direct_bilinear import load as load_direct
from lite.training.numpy_gradients import score_head_forward
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.evaluation_metrics import ranking_metrics

def main():
 import sys
 artifact=Path(sys.argv[1]); out=Path(sys.argv[2]); base=Path.home()/'.cache/memorilla-lite/LITE-007L'; ds=PreparedTrainingSet(artifact,allow_validation=True); source=Path('benchmarks/lite/fixtures/lite007k-independent/dataset.json') if ds.metadata['dataset_version'].startswith('lite007k') else Path('benchmarks/lite/fixtures/lite007i/dataset.json'); data=json.loads(source.read_text()); qs={q['query_id']:q for q in data['queries']}; sc={s['scenario_id']:s for s in data['scenarios']}; step=int(sys.argv[3]) if len(sys.argv)>3 else 240; assoc=base/'associative-initial/checkpoint' if step==0 else base/'associative'/f'episode-{step//8-1:03d}'/'checkpoint'; direct=base/'direct-initial/checkpoint' if step==0 else base/'direct'/f'episode-{step//8-1:03d}'/'checkpoint'
 models={'associative':('assoc',assoc),'direct':('direct',direct),'cosine':('cosine',None)}
 loaded={k:(load_model(p)[0] if t=='assoc' else load_direct(p)[0] if t=='direct' else None) for k,(t,p) in models.items()}
 def rows_for(kind,model):
  rows=[]; diag=[]
  for i in range(len(ds)):
   ex=ds.get(i); q=qs[ex['query_id']]; s=sc[ex['scenario_id']]; valid=ex['valid_mask']; M=ex['memory_embeddings'][valid]; eids=[x for x,v in zip(ex['evidence_ids'],valid) if v]; req=[eids.index(x) for x in q['required_evidence_ids'] if x in eids]; qv=ex['query_embedding']
   if kind=='assoc': scores=score_head_forward(model.forward(qv,M,return_latent=True)[1],M,model.state,np.ones(len(M),bool))[0]
   elif kind=='direct': scores=qv@model@M.T
   else: scores=(qv/np.maximum(np.linalg.norm(qv),1e-9))@(M/np.maximum(np.linalg.norm(M,axis=1,keepdims=True),1e-9)).T
   order=np.argsort(-scores,kind='mergesort'); rows.append({'size':str(s['collection_size']),'category':q['category'],'answerable':not bool(q.get('expected_abstention')),'metrics':ranking_metrics([{'ranked_ids':order.tolist(),'required_ids':req,'answerable':not bool(q.get('expected_abstention'))}])})
   if kind=='assoc':
    (o,att),slots=model.forward(qv,M,np.ones(len(M),bool),return_attention=True,return_latent=True); z=slots/np.maximum(np.linalg.norm(slots,axis=1,keepdims=True),1e-9); P=att[-1]; reqset=set(req); diag.append({'entropy':float(np.mean(-np.sum(np.where(P>0,P*np.log(np.maximum(P,1e-12)),0),axis=1))),'required_attention':float(P[:,list(reqset)].sum()/P.shape[0]) if reqset else 0.0,'uniform_required_attention':len(reqset)/len(M) if M.size else 0.0,'slot_cosine':float(np.mean((z@z.T)[np.triu_indices(len(z),1)])),'query_sensitivity':float(np.linalg.norm(slots-model.forward(np.zeros_like(qv),M,np.ones(len(M)),return_latent=True)[1]))})
  groups={}
  for r in rows:
   for g in [r['size'],r['category'],'answerable' if r['answerable'] else 'abstention']:
    groups.setdefault(g,[]).append(r['metrics'])
  summary={g:{k:sum(x[k] for x in vals if x[k] is not None)/len([x for x in vals if x[k] is not None]) if any(x[k] is not None for x in vals) else None for k in ['hit_rate_at_1','recall_at_1','recall_at_5','mrr','multi_evidence_coverage','complete_required_evidence_at_5','abstention_accuracy']} for g,vals in groups.items()}
  return summary,diag
 result={'artifact':str(artifact),'models':{}}
 for name,(kind,_) in models.items():
  metrics,diag=rows_for(kind,loaded[name]); result['models'][name]={'metrics':metrics}
  if kind=='assoc': result['models'][name]['representation']=diag
 out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); print(out)
if __name__=='__main__': main()
