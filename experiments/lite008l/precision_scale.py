"""LITE-008L BF16/Float32 precision and decoder embedding-scale diagnostics."""
from __future__ import annotations
import argparse,gc,json,time
from pathlib import Path
import numpy as np,torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008l.ledger import phase_start,phase_end
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008L';SEED=20261124

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dtype',choices=['bfloat16','float32'],required=True);args=ap.parse_args();dtype=torch.bfloat16 if args.dtype=='bfloat16' else torch.float32;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED);start=time.monotonic();phase=f'precision_{args.dtype}';phase_start(phase,dtype=args.dtype,seed=SEED)
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);load_start=time.monotonic();model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=dtype,low_cpu_mem_usage=False);load_seconds=time.monotonic()-load_start;model.requires_grad_(False);model.eval();model.config.use_cache=False;emb=model.get_input_embeddings();slot_id=int(tok.eos_token_id);base=emb(torch.tensor([[slot_id]])).detach().float()[0,0];query='Question: What is the capital of France? Answer:';p=[slot_id]+tok(query,add_special_tokens=True)['input_ids'];target=tok(' Paris',add_special_tokens=False)['input_ids'];assert len(target)==1
 def loss(slot):
  full=torch.tensor([p]);x=emb(full).detach().float();x[:,0,:]=slot.float();x=x.to(dtype);out=model(inputs_embeds=x,attention_mask=torch.ones_like(full),use_cache=False).logits.float()[:,-1,:];return torch.nn.functional.cross_entropy(out,torch.tensor(target))
 slot=base.clone().requires_grad_(True);l=loss(slot);l.backward();g=slot.grad.detach().clone();gen=torch.Generator().manual_seed(SEED+12);rows=[]
 for di in range(4):
  d=torch.randn(g.shape,generator=gen);d=d/d.norm();row={'direction':di,'analytic':float((g*d).sum()),'epsilons':{}}
  for eps in [.0625,.125,.25,.5,1.,2.]:
   plus=slot.detach()+eps*d;minus=slot.detach()-eps*d;plus_dec=plus.to(dtype);minus_dec=minus.to(dtype);actual_plus=float((plus_dec-base.to(dtype)).float().norm());actual_minus=float((minus_dec-base.to(dtype)).float().norm());
   with torch.no_grad():lp=float(loss(plus));lm=float(loss(minus))
   num=(lp-lm)/(2*eps);row['epsilons'][str(eps)]={'numerical':num,'relative_error':abs(float((g*d).sum())-num)/max(abs(float((g*d).sum())),1e-4),'actual_plus_step_norm':actual_plus,'actual_minus_step_norm':actual_minus,'representable':actual_plus>0 and actual_minus>0}
  rows.append(row)
 w=emb.weight.detach().float();norms=w.norm(dim=1).numpy();rms=float(w.pow(2).mean().sqrt());scale={'vocab_rows':int(w.shape[0]),'hidden_size':int(w.shape[1]),'rms':rms,'mean_abs':float(w.abs().mean()),'norm_quantiles':{str(q):float(np.quantile(norms,q)) for q in [0,.01,.1,.5,.9,.99,1]},'eos_norm':float(w[slot_id].norm()),'zero_norm':0.0,'random_rms_matched_norm':float(torch.randn_like(base).mul(rms).norm()),'dtype':str(dtype)}
 optimized=ROOT.parent/'LITE-008J'/'optimized_embedding.npy';
 if optimized.exists():
  z=torch.from_numpy(np.load(optimized).astype('float32')).flatten();scale['lite008j_optimized_norm']=float(z.norm());scale['lite008j_optimized_rms']=float(z.pow(2).mean().sqrt())
 result={'schema':'LITE-008L-precision-scale-v1','dtype':args.dtype,'decoder_dtype':str(dtype),'parameter_count':sum(p.numel() for p in model.parameters()),'loading_seconds':load_seconds,'loss':float(l.detach()),'gradient_norm':float(g.norm()),'gradient_finite':bool(torch.isfinite(g).all()),'directions':rows,'embedding_scale':scale,'forward_backward_seconds':time.monotonic()-start,'decoder_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'decoder_parameter_grads_absent':all(p.grad is None for p in model.parameters()),'protected_final_holdout_accessed':False};(ROOT/f'precision-{args.dtype}.json').write_text(json.dumps(result,indent=2)+'\n');phase_end(phase,wall_seconds=result['forward_backward_seconds'],gradient_finite=result['gradient_finite']);print(json.dumps({'dtype':args.dtype,'seconds':result['forward_backward_seconds'],'gradient_norm':result['gradient_norm']}))
if __name__=='__main__':main()
