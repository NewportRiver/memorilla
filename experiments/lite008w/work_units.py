"""Interruptible, content-bound LITE-008W teacher work units."""
from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import torch
from experiments.lite008p.p_experiment import objective,run_forward,ids_for,tokid,MODEL

SOFT_DEADLINE=240.0
HARD_DEADLINE=300.0

def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def canonical(obj): return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic_json(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2)+'\n'); os.replace(tmp,path)
def atomic_torch(path,obj):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def content_identity(row,manifest_sha,encoder_identity,decoder_identity,token_ids,config):
 payload={'manifest_sha256':manifest_sha,'partition':row['partition'],'entity':row['entity'],'case_id':row['id'],'target':row['target'],'evidence':row['evidence'],'query':row['query'],'candidate_answers':[row['answer_a'],row['answer_b']],'candidate_token_ids':token_ids,'factual_direction':row['target'],'encoder_identity':encoder_identity,'decoder_identity':decoder_identity,'teacher_generation_config':config}
 return canonical(payload),payload
def run_direction(row, target, manifest_sha, encoder_identity, outdir, max_updates=16, soft_deadline=SOFT_DEADLINE):
 torch.set_num_threads(1);torch.set_num_interop_threads(1); started=time.monotonic(); outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
 tok=torch.load if False else None
 from transformers import AutoModelForCausalLM,AutoTokenizer
 t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True); m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); m.requires_grad_(False);m.eval()
 a=tokid(t,row['answer_a']);b=tokid(t,row['answer_b']); ids=ids_for(t,row['query'],1); eos=m.get_input_embeddings()(torch.tensor([[t.eos_token_id]])).detach().float()[0,0]; config={'updates':max_updates,'lr':.05,'objective':'contrastive','init':'eos','slots':1,'dtype':'bfloat16_decoder_float32_slot','soft_deadline_seconds':soft_deadline}; dec={'model_path':str(MODEL),'model_config_sha256':sha(MODEL/'config.json'),'tokenizer_sha256':sha(MODEL/'tokenizer.json')}; ident,payload=content_identity(row,manifest_sha,encoder_identity,dec,[a,b],config); ck=outdir/(ident+'.checkpoint.pt'); art=outdir/(ident+'.teacher.pt'); meta=outdir/(ident+'.json'); slot=eos.reshape(1,-1).clone().requires_grad_(True); opt=torch.optim.Adam([slot],lr=.05); trajectory=[]; completed=0
 if ck.exists():
  state=torch.load(ck,map_location='cpu');
  if state.get('identity')!=ident or state.get('config')!=config: raise RuntimeError('incompatible checkpoint')
  slot=state['slot'].float().requires_grad_(True); opt=torch.optim.Adam([slot],lr=.05); opt.load_state_dict(state['optimizer']); trajectory=state['trajectory']; completed=state['completed_updates']
 for u in range(completed,max_updates):
  if time.monotonic()-started>=soft_deadline: break
  pre=objective(m,ids,slot,a,b,target=='A','contrastive'); g=pre['grad']; slot.grad=g; torch.nn.utils.clip_grad_norm_([slot],1.0); opt.step(); opt.zero_grad(set_to_none=True); post=run_forward(m,ids,slot.detach(),a,b); completed=u+1; trajectory.append({'update':completed,'pre_loss':pre['loss'],'pre_margin':pre['margin'],'post_margin':post['margin'],'post_nll_a':post['nll_a'],'post_nll_b':post['nll_b'],'slot_norm':float(slot.detach().norm()),'elapsed_seconds':time.monotonic()-started}); atomic_torch(ck,{'identity':ident,'payload':payload,'config':config,'decoder':dec,'slot':slot.detach(),'optimizer':opt.state_dict(),'trajectory':trajectory,'completed_updates':completed})
 status='COMPLETED' if completed==max_updates else 'INCOMPLETE_DEADLINE'
 if status=='COMPLETED':
  vec=slot.detach().clone(); atomic_torch(art,{'identity':ident,'payload':payload,'config':config,'decoder':dec,'embedding':vec,'trajectory':trajectory,'completed_updates':completed}); reload=torch.load(art,map_location='cpu');
  if reload['identity']!=ident or reload['completed_updates']!=max_updates: raise RuntimeError('reload verification failed')
  atomic_json(meta,{'identity':ident,'payload':payload,'config':config,'decoder':dec,'artifact':str(art),'artifact_sha256':sha(art),'checkpoint':str(ck),'completed_updates':completed,'status':status,'wall_seconds':time.monotonic()-started,'optimizer_updates':completed})
 return {'identity':ident,'status':status,'completed_updates':completed,'artifact':str(art) if status=='COMPLETED' else None,'checkpoint':str(ck),'wall_seconds':time.monotonic()-started}
