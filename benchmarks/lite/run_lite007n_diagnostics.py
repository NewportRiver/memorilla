from __future__ import annotations
import json, hashlib, math, resource, time
from pathlib import Path
import numpy as np
from lite.neural.serialization import load_model
from lite.training.direct_bilinear import load as load_direct
from lite.training.numpy_gradients import score_head_forward, associative_gradients
from lite.training.prepared_dataset import PreparedTrainingSet
from lite.training.evaluation_metrics import ranking_metrics

ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-007L'
INDEP=Path.home()/'.cache/memorilla-lite/LITE-007K/prepared-independent'; LARGE=Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-validation.npz'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadj(p): return json.loads(Path(p).read_text())
def finite_norm(x): return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def cos(a,b):
 a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel(); d=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/d) if d else None

def load_context():
 ds=PreparedTrainingSet(INDEP,allow_validation=True); large=PreparedTrainingSet(LARGE,allow_validation=True)
 assoc=load_model(CACHE/'associative/episode-029/checkpoint')[0]; direct=load_direct(CACHE/'direct/episode-029/checkpoint')[0]; cosine=None
 return ds,large,assoc,direct

def model_scores(kind, model, ex, reverse=False):
 q=ex['query_embedding']; M=ex['memory_embeddings']; valid=ex['valid_mask']; pos=ex['positive_mask']; ids=ex['evidence_ids']
 idx=np.flatnonzero(valid)
 if reverse: idx=idx[::-1]
 Mr=M[idx]; pr=pos[idx]; idr=[ids[i] for i in idx]
 if kind=='assoc':
  (_,latent)=model.forward(q,Mr,np.ones(len(Mr),bool),return_latent=True)
  scores=score_head_forward(latent,Mr,model.state,np.ones(len(Mr),bool))[0]
 elif kind=='direct': scores=q@model@Mr.T
 else: scores=(q/np.maximum(np.linalg.norm(q),1e-9))@(Mr/np.maximum(np.linalg.norm(Mr,axis=1,keepdims=True),1e-9)).T
 return scores,pr,idr,Mr,idx

def margin_row(scores, pos, ids, ex):
 valid=np.isfinite(scores); p=np.flatnonzero(pos & valid); n=np.flatnonzero(~pos & valid)
 if not len(p) or not len(n): return None
 order=np.argsort(-scores,kind='mergesort'); ranked=[ids[i] for i in order]
 bp=float(np.max(scores[p])); bn=float(np.max(scores[n])); hardest=float(bp-np.max(scores[n])); bestp=int(p[np.argmax(scores[p])]); bestn=int(n[np.argmax(scores[n])])
 return {'query_id':ex['query_id'],'scenario_id':ex['scenario_id'],'category':ex['category'],'collection_size':int(valid.sum()),'required_evidence_count':int(pos.sum()),'best_positive_score':bp,'best_negative_score':bn,'best_positive_minus_best_negative':hardest,'hardest_negative_margin':hardest,'score_variance':float(np.var(scores[valid])),'correct_top1':bool(order[0] in p),'ranked_top5':[ids[i] for i in order[:5]],'best_positive_id':ids[bestp],'hardest_negative_id':ids[bestn]}

def summarize(rows):
 out={}
 groups={'all':rows}
 for key in ['collection_size','category','required_evidence_count','correct_top1']:
  for r in rows: groups.setdefault(f'{key}={r[key]}',[]).append(r)
 for k,rs in groups.items():
  out[k]={'n':len(rs),'best_positive_minus_best_negative_mean':float(np.mean([r['best_positive_minus_best_negative'] for r in rs])),'best_positive_minus_best_negative_median':float(np.median([r['best_positive_minus_best_negative'] for r in rs])),'score_variance_mean':float(np.mean([r['score_variance'] for r in rs])),'top1_rate':float(np.mean([r['correct_top1'] for r in rs]))}
 return out

def margins(ds, models):
 result={}
 for name,(kind,model) in models.items():
  rows=[]
  for i in ds.eligible_indices():
   ex=ds.get(i); scores,pos,ids,_,_=model_scores(kind,model,ex); r=margin_row(scores,pos,ids,ex)
   if r: rows.append(r)
  result[name]={'examples':len(rows),'stratified':summarize(rows),'rows':rows}
 return result

def gradients(ds,assoc,direct):
 names=[]; sums={}; sumsq={}; sumvec={}; counts={}; align=[]; sample_vecs=[]; nonfinite=0; start=time.perf_counter()
 for i in ds.eligible_indices():
  ex=ds.get(i); q=ex['query_embedding']; M=ex['memory_embeddings']; valid=ex['valid_mask']; pos=ex['positive_mask']
  loss,scores,g,_=associative_gradients(assoc,q,M,pos,valid)
  flat=np.concatenate([g[k].astype(np.float64).ravel() for k in sorted(g)])
  if not np.isfinite(flat).all(): nonfinite+=1
  for k,v in g.items():
   n=finite_norm(v); sums[k]=sums.get(k,0.0)+n; sumsq[k]=sumsq.get(k,0.0)+n*n; sumvec[k]=sumvec.get(k,np.zeros_like(v,dtype=np.float64))+v.astype(np.float64); counts[k]=counts.get(k,0)+1
  if len(sample_vecs)<20: sample_vecs.append(flat)
  align.append({'query_id':ex['query_id'],'loss':float(loss),'global_gradient_norm':finite_norm(flat),'score_gradient_norm':finite_norm(g['score_slot_w'])+finite_norm(g['score_mem_w'])})
 # direct analytical gradient on first positive and first negative, matching runner's objective
 dnorm=[]
 for i in ds.eligible_indices():
  ex=ds.get(i); q=ex['query_embedding']; M=ex['memory_embeddings']; v=ex['valid_mask']; p=ex['positive_mask']; pi=int(np.flatnonzero(p)[0]); ni=int(np.flatnonzero(v & ~p)[0]); z=float(0.1+q@direct@M[ni]-q@direct@M[pi]); c=1/(1+math.exp(-z)); grad=c*(np.outer(q,M[ni])-np.outer(q,M[pi])); dnorm.append(float(np.linalg.norm(grad)))
 meanflat=np.concatenate([sumvec[k].ravel()/counts[k] for k in sorted(sumvec)]); alignments=[cos(v,meanflat) for v in sample_vecs]
 return {'examples':len(align),'elapsed_seconds':time.perf_counter()-start,'nonfinite_example_count':nonfinite,'associative_tensor_gradient_norms':{k:{'mean':sums[k]/counts[k],'rms':math.sqrt(sumsq[k]/counts[k]),'nonzero_incidence':1.0 if sums[k]>0 else 0.0} for k in sorted(sums)},'associative_global_gradient_norm_mean':float(np.mean([x['global_gradient_norm'] for x in align])),'associative_score_gradient_norm_mean':float(np.mean([x['score_gradient_norm'] for x in align])),'direct_gradient_norm_mean':float(np.mean(dnorm)),'direct_gradient_norm_median':float(np.median(dnorm)),'gradient_direction_alignment_to_associative_mean_sample':{'n':len(alignments),'mean':float(np.mean(alignments)),'min':float(np.min(alignments)),'max':float(np.max(alignments))},'per_example_sample':align[:50],'interpretation':'Raw gradient norms are not optimization-effectiveness proof; gradients were calculated and discarded. Direction alignment is against the aggregate associative gradient over the analyzed corpus, using the first 20 examples as a bounded sample.'}

def slots(ds,assoc):
 rows=[]
 for i in ds.eligible_indices():
  ex=ds.get(i); q=ex['query_embedding']; M=ex['memory_embeddings']; valid=ex['valid_mask']; pos=ex['positive_mask']; ids=ex['evidence_ids']; idx=np.flatnonzero(valid); Mr=M[idx]; pr=pos[idx]
  (out,att),latent=assoc.forward(q,Mr,np.ones(len(Mr),bool),return_attention=True,return_latent=True); P=att[-1]; z=latent/np.maximum(np.linalg.norm(latent,axis=1,keepdims=True),1e-9); ent=float(np.mean(-np.sum(np.where(P>0,P*np.log(np.maximum(P,1e-12)),0),axis=1))); req=np.flatnonzero(pr); enrich=float((P[:,req].sum()/P.shape[0])/(len(req)/len(pr))) if len(req) else None; cov=[float(P[s,req].sum()) if len(req) else 0.0 for s in range(P.shape[0])]; zeroq=np.zeros_like(q); (_,la0)=assoc.forward(zeroq,Mr,np.ones(len(Mr),bool),return_latent=True); sens=float(np.linalg.norm(latent-la0)); rows.append({'query_id':ex['query_id'],'category':ex['category'],'collection_size':int(len(Mr)),'required_evidence_count':int(pr.sum()),'attention_entropy':ent,'required_attention_enrichment_vs_uniform':enrich,'slot_required_attention_coverage':cov,'slot_cosine_mean':float(np.mean((z@z.T)[np.triu_indices(len(z),1)])),'query_conditioning_sensitivity':sens})
 return {'examples':len(rows),'aggregate':{'attention_entropy_mean':float(np.mean([r['attention_entropy'] for r in rows])),'enrichment_mean':float(np.mean([r['required_attention_enrichment_vs_uniform'] for r in rows if r['required_attention_enrichment_vs_uniform'] is not None])),'slot_cosine_mean':float(np.mean([r['slot_cosine_mean'] for r in rows])),'query_sensitivity_mean':float(np.mean([r['query_conditioning_sensitivity'] for r in rows]))},'rows':rows}

def dataset_audit():
 indep=loadj(ROOT/'benchmarks/lite/fixtures/lite007k-independent/dataset.json'); groups={}
 import re
 def norm(s): return re.sub(r'\d+','<N>',re.sub(r'\s+',' ',s.lower()).strip())
 for r in indep['records']: groups.setdefault(norm(r['content']),[]).append(r)
 collisions=[v for v in groups.values() if len(v)>1]
 scenarios={s['scenario_id']:s for s in indep['scenarios']}; queries=indep['queries'];
 return {'training_artifact':loadj(Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-train.json'),'independent_artifact':loadj(Path.home()/'.cache/memorilla-lite/LITE-007K/prepared-independent.json'),'larger_artifact':loadj(Path.home()/'.cache/memorilla-lite/LITE-007J/prepared-validation.json'),'independent_dataset':{'scenarios':len(indep['scenarios']),'queries':len(queries),'records':len(indep['records']),'holdout_included':indep.get('holdout_included'),'normalized_collision_group_count':len(collisions),'collision_group_sizes':[len(x) for x in collisions],'expected_abstention_count':sum(bool(q.get('expected_abstention')) for q in queries),'required_evidence_counts':{str(n):sum(len(q['required_evidence_ids'])==n for q in queries) for n in sorted(set(len(q['required_evidence_ids']) for q in queries))}},'claim_boundary':'Distribution similarity is not labeled leakage; final holdout was not accessed.'}

def main():
 start_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; ds,large,assoc,direct=load_context(); models={'associative':('assoc',assoc),'direct':('direct',direct),'cosine':('cosine',None)}
 # Use independent for primary margins and both artifacts for separate corpus summaries.
 m_ind=margins(ds,models); m_large=margins(large,models); g=gradients(ds,assoc,direct); s=slots(ds,assoc); da=dataset_audit()
 evidence={'baseline_commit':'e8b4e60','dataset_identity':ds.metadata['content_sha256'],'large_dataset_identity':large.metadata['content_sha256'],'encoder_manifest_sha256':ds.metadata['encoder_manifest_sha256'],'checkpoints':{'associative':sha(CACHE/'associative/episode-029/checkpoint.npz'),'direct':sha(CACHE/'direct/episode-029/checkpoint.npz')},'holdout_accessed':False,'checkpoint_replacement':False,'optimizer_updates':0,'resource':{'ru_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'ru_maxrss_delta_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-start_rss,'threads':1,'computational_subprocesses':1}}
 (AUD/'LITE-007N-EVIDENCE-INTEGRITY.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n'); (AUD/'LITE-007N-SCORE-MARGINS.json').write_text(json.dumps({'independent':m_ind,'larger_validation':m_large},indent=2,sort_keys=True)+'\n'); (AUD/'LITE-007N-GRADIENT-ATTRIBUTION.json').write_text(json.dumps(g,indent=2,sort_keys=True)+'\n'); (AUD/'LITE-007N-SLOT-DIAGNOSTICS.json').write_text(json.dumps(s,indent=2,sort_keys=True)+'\n'); (AUD/'LITE-007N-DATASET-SHORTCUT-AUDIT.json').write_text(json.dumps(da,indent=2,sort_keys=True)+'\n'); print('diagnostics complete')
if __name__=='__main__': main()
