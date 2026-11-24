"""Reduced genuine-component BF16/FP32 gradient reference; not full decoder evidence."""
from __future__ import annotations
import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008l.ledger import record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008M'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();record('component_reference','started',components=['input_embedding','rms_norm','tied_output_projection'])
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();emb=m.get_input_embeddings();normw=m.model.norm.weight.detach();outw=emb.weight.detach();sid=int(tok.eos_token_id);ids=torch.tensor([[sid]]);target=torch.tensor(tok(' Paris',add_special_tokens=False)['input_ids']);base=emb(ids).detach().float()[0,0];eps=1e-5;counts={'torch.bfloat16':{'forward':0,'backward':0},'torch.float32':{'forward':0,'backward':0}}
 def reduced(slot,dtype):
  x=emb(ids).detach().to(dtype);x[:,0,:]=slot.to(dtype);w=normw.to(dtype);ow=outw.to(dtype);y=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps);y=y*w;return torch.nn.functional.linear(y[:,-1,:],ow).float()
 def qualify(dtype):
  s=base.clone().requires_grad_(True);logits=reduced(s,dtype);counts[str(dtype)]['forward']+=1;loss=torch.nn.functional.cross_entropy(logits,target);loss.backward();g=s.grad.detach().clone();counts[str(dtype)]['backward']+=1;rows=[]
  for di in range(4):
   d=torch.randn(g.shape,generator=torch.Generator().manual_seed(20261124+di));d=d/d.norm();row={'direction':di,'analytic':float((g*d).sum()),'epsilons':{}}
   for e in [.0625,.125,.25,.5,1.]:
    with torch.no_grad():lp=torch.nn.functional.cross_entropy(reduced(s.detach()+e*d,dtype),target);lm=torch.nn.functional.cross_entropy(reduced(s.detach()-e*d,dtype),target);counts[str(dtype)]['forward']+=2
    num=float((lp-lm)/(2*e));ana=row['analytic'];row['epsilons'][str(e)]={'numerical':num,'absolute_error':abs(ana-num),'relative_error':abs(ana-num)/max(abs(ana),1e-4)}
   rows.append(row)
  return {'loss':float(loss.detach()),'gradient':g,'gradient_norm':float(g.norm()),'directions':rows}
 bf=qualify(torch.bfloat16);fp=qualify(torch.float32);bg=bf['gradient'];fg=fp['gradient'];cos=float(torch.nn.functional.cosine_similarity(bg,fg,dim=0));sign=float((torch.sign(bg)==torch.sign(fg)).float().mean());ratio=float(bg.norm()/fg.norm());
 useful={}
 for label,direction in [('negative_bf16',-bg),('negative_fp32',-fg),('positive_bf16',bg),('random',torch.randn(bg.shape,generator=torch.Generator().manual_seed(9090)))]:
  direction=direction/direction.norm();useful[label]={}
  for step in [.0625,.125,.25]:
   vals=[]
   for dtype in [torch.bfloat16,torch.float32]: vals.append(float(torch.nn.functional.cross_entropy(reduced(base+step*direction,dtype),target)))
   useful[label][str(step)]={'bf16_loss':vals[0],'fp32_loss':vals[1]}
 result={'schema':'LITE-008M-component-reference-v1','status':'component_only','components':['input_embedding','rms_norm','tied_output_projection'],'full_transformer_layers_included':False,'bf16':{k:v for k,v in bf.items() if k!='gradient'},'float32':{k:v for k,v in fp.items() if k!='gradient'},'gradient_comparison':{'cosine':cos,'norm_ratio_bf16_over_fp32':ratio,'coordinate_sign_agreement':sign},'gradient_usefulness':useful,'counts':counts,'wall_seconds':time.monotonic()-start,'decoder_parameters_frozen':all(not p.requires_grad for p in m.parameters()),'protected_final_holdout_accessed':False};(ROOT/'component-reference-results.json').write_text(json.dumps(result,indent=2)+'\n');record('component_reference','completed',wall_seconds=result['wall_seconds'],forward_count=sum(x['forward'] for x in counts.values()),backward_count=sum(x['backward'] for x in counts.values()));print(json.dumps({'cosine':cos,'norm_ratio':ratio,'sign_agreement':sign,'seconds':result['wall_seconds']}))
if __name__=='__main__':main()
