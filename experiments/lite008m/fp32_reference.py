"""Complete FP32 frozen-decoder numerical reference using memory-efficient loading."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008l.ledger import record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008M'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();record('fp32_reference','started',dtype='float32',low_cpu_mem_usage=True);tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);load=time.monotonic();m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,low_cpu_mem_usage=True);load_seconds=time.monotonic()-load;m.requires_grad_(False);m.eval();m.config.use_cache=False;emb=m.get_input_embeddings();slot_id=int(tok.eos_token_id);query='Question: What is the capital of France? Answer:';q=tok(query,add_special_tokens=True)['input_ids'];ids=torch.tensor([[slot_id]+q]);target=torch.tensor(tok(' Paris',add_special_tokens=False)['input_ids']);ordinary=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float();x=emb(ids).detach().float();slot=x[:,0,:].clone().requires_grad_(True);x[:,0,:]=slot;embedded=m(inputs_embeds=x,attention_mask=torch.ones_like(ids),use_cache=False).logits.float();forward_count=2;maxdiff=float((ordinary-embedded).abs().max());loss=torch.nn.functional.cross_entropy(embedded[:,-1,:],target);loss.backward();g=slot.grad.detach().clone();backward_count=1;rows=[]
 for di in range(4):
  d=torch.randn(g.shape,generator=torch.Generator().manual_seed(20261124+di));d=d/d.norm();row={'direction':di,'analytic':float((g*d).sum()),'epsilons':{}}
  for eps in [.0625,.125,.25,.5,1.,2.]:
   with torch.no_grad():
    for sign in (1.,-1.):
     z=slot.detach()+sign*eps*d;xx=emb(ids).detach().float();xx[:,0,:]=z;lo=m(inputs_embeds=xx,attention_mask=torch.ones_like(ids),use_cache=False).logits.float();val=torch.nn.functional.cross_entropy(lo[:,-1,:],target);row['epsilons'].setdefault(str(eps),{})['plus' if sign>0 else 'minus']=float(val);forward_count+=1
   e=row['epsilons'][str(eps)];num=(e['plus']-e['minus'])/(2*eps);ana=row['analytic'];e.update({'numerical':num,'absolute_error':abs(ana-num),'relative_error':abs(ana-num)/max(abs(ana),1e-4)})
  rows.append(row)
 result={'schema':'LITE-008M-fp32-reference-v1','status':'completed','dtype':'float32','low_cpu_mem_usage':True,'loading_seconds':load_seconds,'total_seconds':time.monotonic()-start,'forward_count':forward_count,'backward_count':backward_count,'ordinary_vs_external_max_logit_diff':maxdiff,'loss':float(loss.detach()),'gradient_norm':float(g.norm()),'gradient_finite':bool(torch.isfinite(g).all()),'directions':rows,'decoder_parameters_frozen':all(not p.requires_grad for p in m.parameters()),'decoder_parameter_grads_absent':all(p.grad is None for p in m.parameters()),'protected_final_holdout_accessed':False};(ROOT/'fp32-reference-results.json').write_text(json.dumps(result,indent=2)+'\n');record('fp32_reference','completed',forward_count=forward_count,backward_count=backward_count,wall_seconds=result['total_seconds']);print(json.dumps({'seconds':result['total_seconds'],'forward_count':forward_count,'gradient_norm':result['gradient_norm'],'maxdiff':maxdiff}))
if __name__=='__main__':main()
