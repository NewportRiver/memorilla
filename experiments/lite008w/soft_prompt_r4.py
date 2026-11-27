import json,time,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import objective,run_forward,ids_for,tokid,MODEL
from experiments.lite008w.work_units import atomic_torch,atomic_json,sha
ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-008W-R4'; CACHE.mkdir(parents=True,exist_ok=True)
MAN=AUD/'LITE-008W-R2-DATASET-MANIFEST.json'; ELIG=['wtrain_00','wtrain_01','wtrain_02','wtrain_04','wtrain_05']
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1); start=time.monotonic(); rows={x['id']:x for x in json.loads(MAN.read_text())['rows']}; tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True); model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); model.requires_grad_(False);model.eval(); eos=model.get_input_embeddings()(torch.tensor([[tok.eos_token_id]])).detach().float()[0,0]; cfg={'updates':32,'lr':.05,'objective':'contrastive','init':'eos','slots':1,'dtype':'bfloat16_decoder_float32_slot','memory_independent':True,'balanced_directions':True,'training_entities':ELIG,'manifest_sha256':hashlib.sha256(MAN.read_bytes()).hexdigest(),'soft_deadline_seconds':240}; ident=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(); ck=CACHE/(ident+'.checkpoint.pt'); art=CACHE/(ident+'.soft_prompt.pt'); slot=eos.reshape(1,-1).clone().requires_grad_(True); opt=torch.optim.Adam([slot],lr=.05); traj=[]; completed=0
 if ck.exists():
  s=torch.load(ck,map_location='cpu');
  if s.get('identity')!=ident or s.get('config')!=cfg: raise RuntimeError('incompatible checkpoint')
  slot=s['slot'].float().requires_grad_(True);opt=torch.optim.Adam([slot],lr=.05);opt.load_state_dict(s['optimizer']);traj=s['trajectory'];completed=s['completed_updates']
 # exactly balanced: 16 A and 16 B, deterministic training-row cycle
 schedule=[(ELIG[i%len(ELIG)],'A' if i%2==0 else 'B') for i in range(32)]
 for u in range(completed,32):
  if time.monotonic()-start>=240: break
  cid,target=schedule[u]; r=rows[cid]; ids=ids_for(tok,r['query'],1); a=tokid(tok,r['answer_a']);b=tokid(tok,r['answer_b']); pre=objective(model,ids,slot,a,b,target=='A','contrastive');slot.grad=pre['grad'];torch.nn.utils.clip_grad_norm_([slot],1.0);opt.step();opt.zero_grad(set_to_none=True);post=run_forward(model,ids,slot.detach(),a,b);completed=u+1;traj.append({'update':completed,'case_id':cid,'target':target,'pre_loss':pre['loss'],'pre_margin':pre['margin'],'post_margin':post['margin'],'post_nll_a':post['nll_a'],'post_nll_b':post['nll_b'],'slot_norm':float(slot.detach().norm()),'elapsed_seconds':time.monotonic()-start});atomic_torch(ck,{'identity':ident,'config':cfg,'slot':slot.detach(),'optimizer':opt.state_dict(),'trajectory':traj,'completed_updates':completed})
 status='COMPLETED' if completed==32 else 'INCOMPLETE_DEADLINE'
 if status=='COMPLETED':
  atomic_torch(art,{'identity':ident,'config':cfg,'prompt':slot.detach(),'trajectory':traj,'completed_updates':completed}); re=torch.load(art,map_location='cpu'); assert re['identity']==ident and re['completed_updates']==32
  atomic_json(CACHE/(ident+'.json'),{'identity':ident,'config':cfg,'artifact':str(art),'artifact_sha256':sha(art),'checkpoint':str(ck),'completed_updates':completed,'status':status,'wall_seconds':time.monotonic()-start,'optimizer_updates':32})
  print(json.dumps({'status':status,'identity':ident,'artifact':str(art),'updates':completed,'seconds':time.monotonic()-start}))
 else: raise RuntimeError(f'soft prompt incomplete: {completed}/32')
if __name__=='__main__':main()
