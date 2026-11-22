import json, math, resource, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from lite.neural.serialization import load_model
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.numpy_gradients import associative_gradients, pairwise_loss_and_score_gradient, score_head_forward

ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-007L'; PREP=Path.home()/'.cache/memorilla-lite/LITE-007K/prepared-independent'
def clone_scale(base,scale):
 return type(base)(replace(base.config,attention_logit_scale=scale),state={k:v.copy() for k,v in base.state.items()})
def metrics(ds,model):
 margins=[]; enrich=[]; ent=[]; spread=[]; hits=[]; start=time.perf_counter()
 for i in ds.eligible_indices():
  ex=ds.get(i); q=ex['query_embedding']; M=ex['memory_embeddings']; v=ex['valid_mask']; p=ex['positive_mask']; idx=np.flatnonzero(v); M=M[idx]; p=p[idx]
  (out,att),lat=model.forward(q,M,np.ones(len(M),bool),return_attention=True,return_latent=True); scores=score_head_forward(lat,M,model.state,np.ones(len(M),bool))[0]; order=np.argsort(-scores,kind='mergesort'); pos=np.flatnonzero(p); neg=np.flatnonzero(~p); margins.append(float(np.max(scores[pos])-np.max(scores[neg]))); hits.append(bool(order[0] in pos)); P=att[-1]; req=np.flatnonzero(p); enrich.append(float((P[:,req].sum()/P.shape[0])/(len(req)/len(p)))); ent.append(float(np.mean(-np.sum(P*np.log(np.maximum(P,1e-12)),axis=1)))); logits_spread=[]; s=model.state; c=model.config; slots=s['slots']+(q@s['query_w']+s['query_b'])[None,:]; wk=M@s['mem_k_w']+s['mem_k_b'];wv=M@s['mem_v_w']+s['mem_v_b'];slots=slots+((wk+wv).mean(axis=0))[None,:]
  for bi in range(c.blocks):
   z=slots; z=(z-z.mean(axis=-1,keepdims=True))/np.sqrt(((z-z.mean(axis=-1,keepdims=True))**2).mean(axis=-1,keepdims=True)+1e-5)*s[f'b{bi}.ln1_g']+s[f'b{bi}.ln1_b'];qh=(z@s[f'b{bi}.q_w']+s[f'b{bi}.q_b']).reshape(c.slots,c.heads,-1);kh=(M@s[f'b{bi}.k_w']+s[f'b{bi}.k_b']).reshape(-1,c.heads,qh.shape[-1]); logits=c.attention_logit_scale*np.einsum('shd,nhd->shn',qh,kh)/np.sqrt(qh.shape[-1]);logits_spread.append(float(np.std(logits)))
  spread.append(float(np.mean(logits_spread)))
 return {'examples':len(margins),'hit_at_1':float(np.mean(hits)),'mean_margin':float(np.mean(margins)),'median_margin':float(np.median(margins)),'mean_required_evidence_enrichment':float(np.mean(enrich)),'mean_attention_entropy':float(np.mean(ent)),'mean_attention_logit_spread':float(np.mean(spread)),'wall_seconds':time.perf_counter()-start}
def fd_check(ds,base,scale):
 ex=ds.get(ds.eligible_indices()[0]); q=ex['query_embedding']; M=ex['memory_embeddings']; p=ex['positive_mask'];v=ex['valid_mask']; m=clone_scale(base,scale); _,_,g,_=associative_gradients(m,q,M,p,v); name='b0.q_w'; d=np.random.default_rng(123).standard_normal(m.state[name].shape).astype('f4');d/=np.float32(np.linalg.norm(d));ps=max(float(np.sqrt(np.mean(m.state[name]**2))),1e-3);eps=.5;delta=d*np.float32(eps*ps)
 def loss(x):
  _,lat=x.forward(q,M,v,return_latent=True);sc=score_head_forward(lat,M,x.state,v)[0];return float(pairwise_loss_and_score_gradient(sc,p,v)[0])
 plus=clone_scale(m,scale);minus=clone_scale(m,scale);plus.state[name]+=delta;minus.state[name]-=delta;num=1e8*(loss(plus)-loss(minus))/(2*eps);ana=1e8*float(np.sum(g[name]*d)*ps);return {'scale':scale,'tensor':name,'analytical':ana,'numerical':num,'relative_error':abs(num-ana)/max(abs(ana),1e-2),'tolerance':.25}
def main():
 rss0=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss;ds=PreparedTrainingSet(PREP,allow_validation=True);base=load_model(CACHE/'associative/episode-029/checkpoint')[0];scales=[0.5,1.0,2.0,4.0]; results={str(x):metrics(ds,clone_scale(base,x)) for x in scales}; a=clone_scale(base,1.0); maxdiff=0.0
 for i in ds.eligible_indices()[:32]:
  ex=ds.get(i); o=base.forward(ex['query_embedding'],ex['memory_embeddings'],ex['valid_mask'],return_attention=True,return_latent=True); n=a.forward(ex['query_embedding'],ex['memory_embeddings'],ex['valid_mask'],return_attention=True,return_latent=True); maxdiff=max(maxdiff,float(np.max(np.abs(o[0][0]-n[0][0]))),float(np.max(np.abs(o[1]-n[1]))))
 out={'baseline_commit':'dd8bec3','scales':scales,'scale_1_reproduction_max_abs_difference':maxdiff,'results':results,'derivative_checks':[fd_check(ds,base,x) for x in scales],'resource':{'ru_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'ru_maxrss_delta_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-rss0,'threads':1,'computational_subprocesses':1},'holdout_accessed':False,'optimizer_updates':0,'interpretation':'Frozen diagnostic sweep only; no scale was selected as an unbiased final estimate.'}; (AUD/'LITE-007Q-FEASIBILITY.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');ds.close();print(json.dumps({'maxdiff':maxdiff,'results':results,'fd':out['derivative_checks'],'resource':out['resource']},indent=2))
if __name__=='__main__':main()
