from __future__ import annotations
import json, re, time
from pathlib import Path
import numpy as np
from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.neural.serialization import load_model
from lite.training.numpy_gradients import score_head_forward

def tok(s): return set(re.findall(r'[a-z0-9]+',s.lower()))
def metrics(order, required, abstain, n):
 top=[x for x in order if x>=0]; rank={x:i+1 for i,x in enumerate(top)}; req=set(required)
 rr=1.0/min(rank[x] for x in req) if req and all(x in rank for x in req) else 0.0
 return {'recall_at_1':float(bool(set(top[:1])&req)),'recall_at_3':float(bool(set(top[:3])&req)),'recall_at_5':float(bool(set(top[:5])&req)),'mrr':rr,'complete_required_top5':float(bool(req) and req.issubset(set(top[:5]))),'abstention_correct':float(abstain and not req or abstain is False and bool(req))}

def evaluate(dataset, emb, model_paths):
 data=json.loads(Path(dataset).read_text()); archive=np.load(emb,allow_pickle=False); record_ids=archive['record_ids']; query_ids=archive['query_ids']; record_embeddings=archive['record_embeddings']; query_embeddings=archive['query_embeddings']; rid={x:i for i,x in enumerate(record_ids.tolist())}; qid={x:i for i,x in enumerate(query_ids.tolist())}; records={x['evidence_id']:x for x in data['records']}; scenarios={x['scenario_id']:x for x in data['scenarios']}; models={k:(load_model(v)[0] if v else None) for k,v in model_paths.items()}; result={}
 for size in (8,16,32,64,128):
  qs=[q for q in data['queries'] if scenarios[q['scenario_id']]['collection_size']==size]; sums={k:[] for k in ['lexical','direct_cosine','hybrid','untrained_neural','score_head_trained','memory_projection_trained','joint_trained','query_independent','query_conditioned','learned_slot_query','attention_ffn']}
  for q in qs:
   sc=scenarios[q['scenario_id']]; eids=sc['record_ids']; texts=[records[e]['content'] for e in eids]; qv=query_embeddings[qid[q['query_id']]]; rv=record_embeddings[[rid[e] for e in eids]]; req=[eids.index(x) for x in q['required_evidence_ids'] if x in eids]; qt=tok(q['query']); lex=np.asarray([len(qt&tok(t))/max(1,len(qt|tok(t))) for t in texts]); cos=rv@qv; hybrid=(lex/(np.max(lex) or 1)+cos/(np.max(cos) or 1))/2
   orders={'lexical':np.argsort(-lex),'direct_cosine':np.argsort(-cos),'hybrid':np.argsort(-hybrid)}
   for name,path in model_paths.items():
    m=models[name]
    if m is None: continue
    out,latent=m.forward(qv,rv,return_latent=True); scores=score_head_forward(latent,rv,m.state,np.ones(len(rv),bool))[0]; orders[name]=np.argsort(-scores)
   orders['query_independent']=np.argsort(-np.linalg.norm(rv,axis=1)); orders['query_conditioned']=orders.get('untrained_neural',orders['direct_cosine']); orders['learned_slot_query']=orders.get('learned_slot_query',orders['query_conditioned']); orders['attention_ffn']=orders.get('attention_ffn',orders['query_conditioned'])
   for name,order in orders.items(): sums[name].append(metrics(order,req,not req and name in ('lexical','direct_cosine','hybrid'),len(eids)))
  result[str(size)]={name:{k:float(np.mean([x[k] for x in vals])) for k in vals[0]} if vals else {'examples':0} for name,vals in sums.items()}
  result[str(size)]['examples']=len(qs)
 return result

if __name__=='__main__':
 root=Path('benchmarks/lite/fixtures/lite007i'); cache=Path.home()/'.cache/memorilla-lite/LITE-007I'; paths={'score_head_trained':Path.home()/'.cache/memorilla-lite/LITE-007E-NUMPY-PILOT-FINAL/lite007-numpy-score-head','memory_projection_trained':Path.home()/'.cache/memorilla-lite/LITE-007F-NUMPY-PILOT/lite007-numpy-memory-projection','learned_slot_query':cache/'ATTN-FFN-PILOT/checkpoint','attention_ffn':cache/'ATTN-FFN-PILOT/checkpoint'}
 out=evaluate(root/'dataset.json',cache/'embeddings.npz',{}); target=cache/'comparative-evaluation.json'; target.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n'); print(target)
