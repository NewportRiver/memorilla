from __future__ import annotations
import torch
import torch.nn.functional as F

def score_latent(model,latent,memory,mask):
 c=model.config
 a=F.linear(latent,model.score_slot_w.T,model.score_slot_b)
 b=F.linear(memory,model.score_mem_w.T,model.score_mem_b)
 a=F.normalize(a,dim=-1); b=F.normalize(b,dim=-1)
 scores=torch.einsum('bsd,bnd->bsn',a,b).amax(dim=1)
 return scores.masked_fill(~mask,-torch.inf)

def pairwise_loss(scores,positive_mask,valid_mask,margin=0.0):
 losses=[]
 for s,p,v in zip(scores,positive_mask,valid_mask):
  pos=s[p & v]; neg=s[(~p) & v]
  if pos.numel()==0 or neg.numel()==0: continue
  losses.append(F.softplus(margin + neg[:,None] - pos[None,:]).mean())
 if not losses: return scores.sum()*0.0
 return torch.stack(losses).mean()

def objective_spec():
 return {'primary':'multi-positive pairwise logistic ranking','margin':0.0,'coverage_loss':0.0,'diversity_loss':0.0,'abstention_loss':0.0,'label_boundary':'authoritative evidence IDs only; policy-ineligible and holdout labels excluded'}
