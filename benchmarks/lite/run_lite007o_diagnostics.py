from __future__ import annotations
import copy, hashlib, json, math, resource, time
from pathlib import Path
import numpy as np
from lite.neural.serialization import load_model
from lite.training.numpy_gradients import associative_gradients, score_head_forward, pairwise_loss_and_score_gradient
from lite.training.prepared_dataset import PreparedTrainingSet

ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-007L'; PREP=Path.home()/'.cache/memorilla-lite/LITE-007K/prepared-independent'
LR=0.001; CLIP=1.0

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(x): return float(np.linalg.norm(np.asarray(x,dtype=np.float64)))
def rms(x): return float(np.sqrt(np.mean(np.square(np.asarray(x,dtype=np.float64)))))
def cosine(a,b):
 a=np.asarray(a,dtype=np.float64).ravel(); b=np.asarray(b,dtype=np.float64).ravel(); d=np.linalg.norm(a)*np.linalg.norm(b); return float(np.dot(a,b)/d) if d else 0.0

def loss_for(model,ex):
 q=ex['query_embedding']; M=ex['memory_embeddings']; valid=ex['valid_mask']; pos=ex['positive_mask'];
 _,latent=model.forward(q,M,valid,return_latent=True); scores=score_head_forward(latent,M,model.state,valid)[0]; loss,_=pairwise_loss_and_score_gradient(scores,pos,valid); return float(loss),scores

def copy_model(model):
 from lite.neural.numpy_model import NumpyAssociativeMemory
 return NumpyAssociativeMemory(model.config,state={k:v.copy() for k,v in model.state.items()})

def normalized(ds,model):
 sums={}; sumsq={}; gsum={}; gsum2={}; count=0; align={}; sample=[]; global_norms=[]; grad_vectors={}
 names=sorted(k for k in model.state if k not in ('out_w','out_b'))
 for i in ds.eligible_indices():
  ex=ds.get(i); loss,scores,g,_=associative_gradients(model,ex['query_embedding'],ex['memory_embeddings'],ex['positive_mask'],ex['valid_mask']); count+=1
  gn=math.sqrt(sum(np.square(v.astype(np.float64)).sum() for v in g.values())); scale=min(1.0,CLIP/max(gn,1e-12)); global_norms.append(gn)
  for k in names:
   v=g[k].astype(np.float64); sums[k]=sums.get(k,np.zeros_like(v))+v; sumsq[k]=sumsq.get(k,np.zeros_like(v))+v*v; n=norm(v); gsum[k]=gsum.get(k,0)+n; gsum2[k]=gsum2.get(k,0)+n*n
  if len(sample)<20: sample.append({k:g[k].copy() for k in names})
 rows={}
 for k in names:
  p=model.state[k]; mean=sums[k]/count; mean_rms=float(np.sqrt(np.mean(mean*mean))); grad_rms=float(np.sqrt(np.mean(sumsq[k]/count))); p_rms=rms(p); p_norm=norm(p); mean_norm=norm(mean); var=float(np.mean(sumsq[k]/count-mean*mean));
  rows[k]={'shape':list(p.shape),'parameters':int(p.size),'raw_gradient_l2_mean':gsum[k]/count,'raw_gradient_l2_rms':math.sqrt(gsum2[k]/count),'gradient_rms_per_parameter':grad_rms,'gradient_component_variance_mean':max(0.0,var),'parameter_rms':p_rms,'parameter_l2_norm':p_norm,'gradient_rms_to_parameter_rms':grad_rms/max(p_rms,1e-30),'mean_gradient_rms_to_parameter_rms':mean_rms/max(p_rms,1e-30),'proposed_update_rms_unclipped':LR*grad_rms,'proposed_update_to_parameter_rms_unclipped':LR*grad_rms/max(p_rms,1e-30),'global_clip_scale_mean':float(np.mean([min(1.0,CLIP/x) for x in global_norms])),'sample_direction_alignment_to_mean':float(np.mean([cosine(s[k],mean) for s in sample]))}
 return {'examples':count,'tensor_count':len(names),'learning_rate':LR,'clip_norm':CLIP,'tensors':rows,'global_gradient_norm_mean':float(np.mean(global_norms)),'global_gradient_norm_median':float(np.median(global_norms)),'global_clip_fraction':float(np.mean(np.array(global_norms)>CLIP)),'interpretation':'Ratios use FP32 parameter RMS. Proposed updates are hypothetical calculations only; no update was applied.'}

def activations(ds,model):
 totals={}; n=0; entropy=[]; spreads=[]; sizes=[]
 for i in ds.eligible_indices():
  ex=ds.get(i); q=ex['query_embedding']; M=ex['memory_embeddings']; mask=ex['valid_mask']; idx=np.flatnonzero(mask); M=M[idx]; n+=1; sizes.append(len(M)); s=model.state; c=model.config; slots=s['slots']+(q@s['query_w']+s['query_b'])[None,:]; wk=M@s['mem_k_w']+s['mem_k_b']; wv=M@s['mem_v_w']+s['mem_v_b']; slots=slots+((wk+wv).mean(axis=0))[None,:]
  for key,x in [('query_embedding',q),('memory_embedding',M),('query_projection',q@s['query_w']+s['query_b']),('memory_key_projection',wk),('memory_value_projection',wv),('initial_slots',slots)]: totals.setdefault(key,[]).append((rms(x),norm(x)))
  for bi in range(c.blocks):
   p=f'b{bi}.'; ln=(slots-slots.mean(axis=-1,keepdims=True))/np.sqrt(((slots-slots.mean(axis=-1,keepdims=True))**2).mean(axis=-1,keepdims=True)+1e-5)*s[p+'ln1_g']+s[p+'ln1_b']; qh=(ln@s[p+'q_w']+s[p+'q_b']).reshape(c.slots,c.heads,-1); kh=(M@s[p+'k_w']+s[p+'k_b']).reshape(-1,c.heads,qh.shape[-1]); vh=(M@s[p+'v_w']+s[p+'v_b']).reshape(-1,c.heads,qh.shape[-1]); logits=np.einsum('shd,nhd->shn',qh,kh)/np.sqrt(qh.shape[-1]); logits=np.where(mask[idx],logits,-np.inf) if False else logits; mx=np.max(logits,axis=-1,keepdims=True); P=np.exp(logits-mx); P/=P.sum(axis=-1,keepdims=True); entropy.append(float(np.mean(-np.sum(P*np.log(np.maximum(P,1e-12)),axis=-1)))); spreads.append(float(np.mean(np.std(logits,axis=-1)))); ctx=np.einsum('shn,nhd->shd',P,vh).reshape(c.slots,c.latent_dim); slots=slots+ctx@s[p+'attn_out_w']+s[p+'attn_out_b']; z=slots-slots.mean(axis=-1,keepdims=True); z=z/np.sqrt((z*z).mean(axis=-1,keepdims=True)+1e-5)*s[p+'ln2_g']+s[p+'ln2_b']; ff1=z@s[p+'ff1_w']+s[p+'ff1_b']; slots=slots+0 # retain post-attention scale before FF residual
   totals.setdefault(f'b{bi}.post_attention_slots',[]).append((rms(slots),norm(slots))); slots=slots+(0.5*ff1*(1+np.tanh(np.sqrt(2/np.pi)*(ff1+0.044715*ff1**3)))@s[p+'ff2_w']+s[p+'ff2_b'])
  totals.setdefault('final_slots',[]).append((rms(slots),norm(slots)))
 return {'examples':n,'eligible_position_counts':sorted(set(sizes)),'activation_scales':{k:{'rms_mean':float(np.mean([x[0] for x in v])),'l2_mean':float(np.mean([x[1] for x in v]))} for k,v in totals.items()},'attention_logits_std_mean':float(np.mean(spreads)),'attention_entropy_mean':float(np.mean(entropy)),'attention_entropy_by_position_count':{str(sz):float(np.mean([entropy[j] for j,x in enumerate(sizes) if x==sz])) for sz in sorted(set(sizes))},'attention_axis':'softmax over eligible evidence positions for each slot and head; reported entropy averages slot/head rows','attention_position_count_explanation':'Entropy near ln(16)=2.77259 occurs for 16 eligible evidence positions and near-uniform attention; it is not an attention axis of 16 slots.'}

def jacobian(ds,model):
 groups={'query_projection':['query_w','query_b'],'key_projection':['b0.q_w','b0.q_b','b0.k_w','b0.k_b','mem_k_w','mem_k_b'],'value_projection':['b0.v_w','b0.v_b','mem_v_w','mem_v_b'],'attention_output':['b0.attn_out_w','b0.attn_out_b'],'latent_slots':['slots'],'final_scoring':['score_slot_w','score_slot_b','score_mem_w','score_mem_b']}; rows={g:[] for g in groups}; fd=[]; eps=1e-2
 for i in ds.eligible_indices()[:40]:
  ex=ds.get(i); _,_,gr,_=associative_gradients(model,ex['query_embedding'],ex['memory_embeddings'],ex['positive_mask'],ex['valid_mask']); base,_=loss_for(model,ex)
  for group,names in groups.items(): rows[group].append({'gradient_l2':math.sqrt(sum(norm(gr[k])**2 for k in names)),'gradient_rms':math.sqrt(sum(np.mean(np.square(gr[k].astype(np.float64))) for k in names)/len(names))})
  for name in ('query_w','mem_k_w','mem_v_w','b0.attn_out_w','slots','score_slot_w'):
   direction=np.random.default_rng(i+17).standard_normal(model.state[name].shape).astype(np.float32); direction/=max(norm(direction),1e-12); mplus=copy_model(model); mminus=copy_model(model); scale=max(rms(model.state[name]),1e-6); delta=direction*np.float32(eps*scale); mplus.state[name]+=delta; mminus.state[name]-=delta; lp,_=loss_for(mplus,ex); lm,_=loss_for(mminus,ex); fd_delta=float((lp-lm)/(2*eps)); expected=float(np.sum(gr[name]*direction)*scale); fd.append({'query_id':ex['query_id'],'tensor':name,'analytical_directional_derivative':expected,'finite_difference_directional_derivative':fd_delta,'relative_error_to_expected':float(abs(fd_delta-expected)/max(abs(expected),1e-8))})
 return {'examples':40,'groups':{k:{'gradient_l2_mean':float(np.mean([x['gradient_l2'] for x in v])),'gradient_rms_mean':float(np.mean([x['gradient_rms'] for x in v]))} for k,v in rows.items()},'finite_difference':fd,'interpretation':'Finite differences perturb copied in-memory states only; checkpoint files are untouched.'}

def main():
 start=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; ds=PreparedTrainingSet(PREP,allow_validation=True); model=load_model(CACHE/'associative/episode-029/checkpoint')[0]; initial=load_model(CACHE/'associative-initial/checkpoint')[0]
 (AUD/'LITE-007O-NORMALIZED-GRADIENTS.json').write_text(json.dumps(normalized(ds,model),indent=2,sort_keys=True)+'\n'); (AUD/'LITE-007O-JACOBIAN-ATTRIBUTION.json').write_text(json.dumps(jacobian(ds,model),indent=2,sort_keys=True)+'\n'); final=activations(ds,model); init=activations(ds,initial); (AUD/'LITE-007O-ATTENTION-ANALYSIS.json').write_text(json.dumps({'initial':init,'final':final,'resource':{'ru_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'ru_maxrss_delta_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss-start,'threads':1,'computational_subprocesses':1}},indent=2,sort_keys=True)+'\n'); ds.close(); print('LITE007O_DIAGNOSTICS_COMPLETE')
if __name__=='__main__': main()
