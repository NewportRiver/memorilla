from __future__ import annotations
import numpy as np

def rank_evidence(scores,evidence_ids,valid_mask=None,top_k=5):
 scores=np.asarray(scores,dtype=np.float32); ids=list(evidence_ids)
 if scores.ndim!=1 or len(ids)!=len(scores): raise ValueError('scores and evidence_ids must align')
 if not np.isfinite(scores).all(): raise ValueError('non-finite evidence score')
 valid=np.ones(len(ids),bool) if valid_mask is None else np.asarray(valid_mask,bool)
 if valid.shape!=(len(ids),): raise ValueError('valid_mask shape mismatch')
 if top_k<1: raise ValueError('top_k must be positive')
 order=sorted((i for i in range(len(ids)) if valid[i]),key=lambda i:(-float(scores[i]),str(ids[i])))[:top_k]
 return [{'evidence_id':ids[i],'score':float(scores[i]),'rank':r+1} for r,i in enumerate(order)]

def score_slots(slots,evidence_embeddings,slot_projection=None,evidence_projection=None,valid_mask=None):
 slots=np.asarray(slots,np.float32); docs=np.asarray(evidence_embeddings,np.float32)
 if slots.ndim!=2 or docs.ndim!=2 or slots.shape[1]!=docs.shape[1] or not np.isfinite(slots).all() or not np.isfinite(docs).all(): raise ValueError('slot/evidence dimensions or values invalid')
 d=slots.shape[1]; sp=np.eye(d,dtype=np.float32) if slot_projection is None else np.asarray(slot_projection,np.float32); ep=np.eye(d,dtype=np.float32) if evidence_projection is None else np.asarray(evidence_projection,np.float32)
 if sp.shape[0]!=d or ep.shape[0]!=d: raise ValueError('projection dimension mismatch')
 a=slots@sp; b=docs@ep; a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-8); b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-8); scores=np.max(a@b.T,axis=0)
 if valid_mask is not None: scores=np.where(np.asarray(valid_mask,bool),scores,-np.inf)
 return scores.astype(np.float32)

def policy_rank(slots,evidence_embeddings,evidence_ids,verifier,collection_id,access_scope,as_of=None,top_k=5,**kwargs):
 raw_valid=set(verifier.catalog.eligible(collection_id,access_scope,as_of))
 base_valid=np.asarray([e in raw_valid for e in evidence_ids],dtype=bool)
 if 'valid_mask' in kwargs: base_valid &= np.asarray(kwargs.pop('valid_mask'),bool)
 raw=score_slots(slots,evidence_embeddings,valid_mask=base_valid,**kwargs); candidates=rank_evidence(np.where(np.isfinite(raw),raw,0),evidence_ids,np.isfinite(raw),top_k); verified=[]; omitted=[]
 for c in candidates:
  packet,reason=verifier.verify(c,access_scope=access_scope,collection_id=collection_id,as_of=as_of)
  if packet is None: omitted.append({'evidence_id':c['evidence_id'],'reason':reason})
  else: verified.append({**c,**packet})
 return {'results':verified,'omitted':omitted,'abstained':not verified}
