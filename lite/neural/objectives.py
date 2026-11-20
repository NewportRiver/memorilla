from __future__ import annotations
import numpy as np

def objective_spec():
 return {'primary':'multi-positive pairwise logistic ranking','optional':['hard-negative weighting','multi-evidence coverage','slot diversity regularization','evidence-selection abstention'],'label_boundary':'positive IDs must resolve to authoritative evidence; unauthorized/future/holdout labels are excluded'}

def multi_positive_pairwise_loss(scores,positive_mask,valid_mask=None,margin=0.0):
 s=np.asarray(scores,np.float32); p=np.asarray(positive_mask,bool)
 if s.ndim!=2 or p.shape!=s.shape or not np.isfinite(s).all(): raise ValueError('scores and positive_mask must be finite [slots,evidence]')
 valid=np.ones(s.shape[1],bool) if valid_mask is None else np.asarray(valid_mask,bool)
 positives=s[:,p[0] & valid] if p.shape[0]==1 else s[p & np.broadcast_to(valid,p.shape)]
 if positives.size==0: return 0.0
 # Pair each positive evidence against all valid negatives. This is a NumPy design check, not training.
 pos=s[:,p.any(axis=0)&valid].mean(axis=0); neg=s[:,(~p.any(axis=0))&valid]
 if neg.size==0:return 0.0
 z=margin-neg[:,:,None]+pos[None,None,:]; return float(np.log1p(np.exp(np.clip(z,-30,30))).mean())

def slot_diversity_penalty(slots):
 x=np.asarray(slots,np.float32); x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8); gram=x@x.T; off=gram-np.eye(x.shape[0],dtype=np.float32); return float((off*off).mean())
