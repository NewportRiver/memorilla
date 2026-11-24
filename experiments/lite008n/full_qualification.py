"""Qualify full 30-layer layer-wise FP32 reference against finite differences and BF16."""
from __future__ import annotations
import json,time,torch
from pathlib import Path
from experiments.lite008n.layerwise_reference import setup,layerwise
from experiments.lite008l.ledger import record
ROOT=Path.home()/'.cache/memorilla-lite/LITE-008N'
CASES=[('Paris','London','What is the capital of France?'),('blue','green','What color is the ocean in this test?'),('green','blue','What color is grass in this test?')]
def prompt(tok,q):return torch.tensor([[int(tok.eos_token_id)]+tok('Question: '+q+' Answer:',add_special_tokens=True)['input_ids']])
def target(tok,a):return torch.tensor(tok(' '+a,add_special_tokens=False)['input_ids'])
def fp_loss(m,ids,slot,tgt):
 x=m.get_input_embeddings()(ids).detach().float();x[:,0,:]=slot;return layerwise(m,x,30,tgt)['loss']
def bf_grad(m,ids,slot,tgt):
 x=m.get_input_embeddings()(ids).detach();s=slot.detach().to(torch.float32).requires_grad_(True);x[:,0,:]=s.to(x.dtype);out=m(inputs_embeds=x,attention_mask=torch.ones_like(ids),use_cache=False).logits.float();loss=torch.nn.functional.cross_entropy(out[:,-1,:],tgt);loss.backward();return float(loss),s.grad.detach()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();record('full_qualification','started',layers=30);tok,m,ids0,tgt0=setup();base=m.get_input_embeddings()(ids0).detach().float()[0,0];base_result=layerwise(m,m.get_input_embeddings()(ids0).detach().float(),30,tgt0);g=base_result['grad'][0,0];rows=[]
 for di in range(4):
  d=torch.randn(g.shape,generator=torch.Generator().manual_seed(20261124+di));d=d/d.norm();ana=float((g*d).sum());row={'direction':di,'analytic':ana,'epsilons':{}}
  for e in [.0625,.125,.25,.5,1.]:
   plus=fp_loss(m,ids0,base+e*d,tgt0);minus=fp_loss(m,ids0,base-e*d,tgt0);num=(plus-minus)/(2*e);row['epsilons'][str(e)]={'numerical':num,'absolute_error':abs(ana-num),'relative_error':abs(ana-num)/max(abs(ana),1e-4)}
  rows.append(row)
 ids=ids0;tgt=tgt0;bf_loss,bfg=bf_grad(m,ids,base,tgt);cos=float(torch.nn.functional.cosine_similarity(g.flatten(),bfg.flatten(),dim=0));ratio=float(bfg.norm()/g.norm());sign=float((torch.sign(g)==torch.sign(bfg)).float().mean());
 usefulness=[]
 for ci,(correct,alternate,q) in enumerate(CASES):
  ids=prompt(tok,q);tc=target(tok,correct);ta=target(tok,alternate);slot=m.get_input_embeddings()(ids).detach().float()[0,0];fr=layerwise(m, m.get_input_embeddings()(ids).detach().float(),30,tc);fg=fr['grad'][0,0];_,bg=bf_grad(m,ids,slot,tc);dirs={'negative_fp32':-fg,'negative_bf16':-bg,'positive_bf16':bg,'random':torch.randn(fg.shape,generator=torch.Generator().manual_seed(9000+ci))};case={'case':ci,'correct':correct,'alternate':alternate,'no_update':{'correct_nll':fp_loss(m,ids,slot,tc),'alternate_nll':fp_loss(m,ids,slot,ta)},'steps':{}}
  for name,d in dirs.items():
   d=d/d.norm();case['steps'][name]={}
   for step in [.0625,.125,.25]:
    cs=fp_loss(m,ids,slot+step*d,tc);alt=fp_loss(m,ids,slot+step*d,ta);case['steps'][name][str(step)]={'correct_nll':cs,'alternate_nll':alt,'margin_correct_minus_alternate':alt-cs}
  usefulness.append(case)
 out={'schema':'LITE-008N-full-qualification-v1','status':'completed','layers':30,'finite_difference_directions':rows,'fp32_gradient_norm':float(g.norm()),'fp32_gradient_finite':bool(torch.isfinite(g).all()),'bf16_loss':bf_loss,'fp32_loss':base_result['loss'],'bf16_fp32_gradient_comparison':{'cosine':cos,'coordinate_sign_agreement':sign,'norm_ratio_bf16_over_fp32':ratio,'relative_l2_error':float((bfg-g).norm()/g.norm()),'forward_loss_difference':abs(bf_loss-base_result['loss'])},'gradient_usefulness':usefulness,'wall_seconds':time.monotonic()-start,'protected_final_holdout_accessed':False};(ROOT/'full-qualification-results.json').write_text(json.dumps(out,indent=2)+'\n');record('full_qualification','completed',wall_seconds=out['wall_seconds'],forward_count=60,backward_count=31);print(json.dumps({'seconds':out['wall_seconds'],'gradient_norm':out['fp32_gradient_norm'],'bf16_cosine':cos,'bf16_sign':sign,'bf16_norm_ratio':ratio}))
if __name__=='__main__':main()
