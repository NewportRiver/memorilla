from __future__ import annotations
import json,time,random
from pathlib import Path
import numpy as np
import torch
from .config import TrainingConfig
from .dataset import load_development
from .objectives import pairwise_loss,score_latent,objective_spec
from .telemetry import rss_bytes,available_ram_bytes
from .evaluate import metrics
from .export import export_torch
from lite.neural.config import NeuralConfig
from lite.neural.torch_model import TorchAssociativeMemory

def _batch(examples):
 n=max((len(e.evidence_ids) for e in examples),default=0); n=max(n,1); B=len(examples); q=np.stack([e.query_embedding for e in examples]); m=np.zeros((B,n,384),np.float32); valid=np.zeros((B,n),bool); pos=np.zeros((B,n),bool)
 for i,e in enumerate(examples):
  k=len(e.evidence_ids); m[i,:k]=e.memory_embeddings; valid[i,:k]=e.valid_mask; pos[i,:k]=e.positive_mask
 return torch.from_numpy(q),torch.from_numpy(m),torch.from_numpy(valid),torch.from_numpy(pos)

def gradient_audit(model,examples,device='cpu'):
 model.zero_grad(set_to_none=True); usable=[e for e in examples if e.positive_mask.any() and e.valid_mask.sum()>e.positive_mask.sum()][:1]
 if not usable: return {'status':'blocked','reason':'no_positive_negative_example'}
 q,m,v,p=_batch(usable); q,m,v,p=q.to(device),m.to(device),v.to(device),p.to(device); _,latent=model(q,m,v,return_latent=True); loss=pairwise_loss(score_latent(model,latent,m,v),p,v); loss.backward(); rows=[]
 for name,param in model.named_parameters(): rows.append({'name':name,'numel':param.numel(),'nonzero':int(param.grad is not None and torch.count_nonzero(param.grad).item()>0),'grad_norm':float(param.grad.norm().item()) if param.grad is not None else 0.0})
 return {'status':'passed','loss':float(loss.item()),'parameters':rows,'zero_gradient_parameters':[x['name'] for x in rows if not x['nonzero']]}

def train_run(root,config=None,updates=None,output_dir=None,device='cpu'):
 cfg=config or TrainingConfig(); updates=updates if updates is not None else cfg.max_updates; output=Path(output_dir or Path(root)/'runs/lite007'); output.mkdir(parents=True,exist_ok=True); torch.set_num_threads(cfg.max_threads); torch.set_num_interop_threads(1); random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
 examples,data=load_development(Path(root),split='train'); val,_=load_development(Path(root),split='validation'); model=TorchAssociativeMemory(NeuralConfig(seed=cfg.seed)).to(device); audit=gradient_audit(model,examples,device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay); start=time.perf_counter(); telemetry=[]; update=0; stop_reason='max_updates'; usable=[e for e in examples if e.positive_mask.any() and e.valid_mask.sum()>e.positive_mask.sum()]
 if audit.get('status')!='passed': return {'status':'blocked','reason':'gradient_audit','gradient_audit':audit}
 while update<updates:
  if time.perf_counter()-start>cfg.max_seconds: stop_reason='time_limit'; break
  if available_ram_bytes() and available_ram_bytes()<cfg.abort_available_ram_bytes: stop_reason='available_ram_abort'; break
  if rss_bytes()>cfg.max_rss_bytes: stop_reason='rss_abort'; break
  batch=[usable[update%len(usable)]]
  q,m,v,p=_batch(batch); q,m,v,p=q.to(device),m.to(device),v.to(device),p.to(device); opt.zero_grad(set_to_none=True); _,latent=model(q,m,v,return_latent=True); loss=pairwise_loss(score_latent(model,latent,m,v),p,v)
  if not torch.isfinite(loss): stop_reason='nonfinite_loss'; break
  loss.backward(); grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)); opt.step(); update+=1
  telemetry.append({'update':update,'loss':float(loss.item()),'grad_norm':grad_norm,'rss_bytes':rss_bytes(),'available_ram_bytes':available_ram_bytes(),'elapsed_seconds':time.perf_counter()-start,'learning_rate':opt.param_groups[0]['lr']})
  if update%max(1,min(10,updates))==0: (output/f'telemetry-{update:05d}.json').write_text(json.dumps(telemetry,sort_keys=True,indent=2)+'\n')
 status='passed_pilot' if update else 'blocked'; result={'status':status,'stop_reason':stop_reason,'updates':update,'epochs':0,'config':cfg.as_dict(),'architecture':NeuralConfig(seed=cfg.seed).as_dict(),'gradient_audit':audit,'telemetry':telemetry,'train_metrics':metrics(model,examples[:min(64,len(examples))]),'validation_metrics':metrics(model,val),'device':device,'peak_rss_bytes':rss_bytes(),'checkpoint':None}
 if update:
  result['checkpoint']=export_torch(model,output/'lite007-pilot',{'repo':'all-MiniLM-L6-v2','revision':'pinned','sha256':'4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474','dimension':384,'contract':'attention-mask-mean-l2-v1'},'trained_pilot_android')
 (output/'run.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n'); return result
