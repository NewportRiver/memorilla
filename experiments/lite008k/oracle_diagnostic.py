"""LITE-008K numerical correction and counterfactual oracle prerequisite."""
from __future__ import annotations
import json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008K';SEED=20261124

def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(SEED);start=time.monotonic();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=False);model.requires_grad_(False);model.eval();model.config.use_cache=False;emb=model.get_input_embeddings();dtype=next(model.parameters()).dtype
 manifest=json.loads((ROOT/'dataset-manifest.json').read_text());rows=manifest['rows']
 def aids(v):
  x=tok(' '+v,add_special_tokens=False)['input_ids']
  if not x: raise AssertionError(v)
  return x
 tokenizations={v:aids(v) for v in manifest['values']}; assert all(len(x)==1 for x in tokenizations.values())
 slot_id=int(tok.eos_token_id); slot_base=emb(torch.tensor([[slot_id]])).detach().float()[0,0]
 def pids(q): return [slot_id]+tok(q,add_special_tokens=True)['input_ids']
 def pair_logits(slot,q):
  p=pids(q); full=torch.tensor([p]);x=emb(full).detach().float();x[:,0,:]=slot.float();x=x.to(dtype);logits=model(inputs_embeds=x,attention_mask=torch.ones_like(full),use_cache=False).logits.float()[:,len(p)-1,:].log_softmax(-1);return logits
 def score(slot,q,a,b):
  lp=pair_logits(slot,q);ia,ib=aids(a)[0],aids(b)[0];la=lp[0,ia];lb=lp[0,ib];return {'a_logprob':float(la.detach()),'b_logprob':float(lb.detach()),'margin':float((la-lb).detach()),'a_nll':float(-la.detach()),'b_nll':float(-lb.detach()),'answer_tokens':aids(a),'alternate_tokens':aids(b)}
 def objective(slot,q,a,b,which):
  lp=pair_logits(slot,q);ia,ib=aids(a)[0],aids(b)[0];la=lp[0,ia];lb=lp[0,ib];target=la if which=='a' else lb;margin=(la-lb) if which=='a' else (lb-la);nll=-target;return nll+0.5*torch.nn.functional.softplus(-margin),float(nll.detach()),float(margin.detach())
 q='What color belongs to entity cobalt?';a='red';b='blue';base=slot_base.clone().requires_grad_(True);loss,_,_=objective(base,q,a,b,'a');loss.backward();g=base.grad.detach().clone();gen=torch.Generator().manual_seed(SEED+20);directions=[]
 for i in range(3):
  d=torch.randn(g.shape,generator=gen);d=d/d.norm();row={'direction':i,'analytic':float((g*d).sum())}
  for eps in [.25,.5,1.,2.]:
   with torch.no_grad(): lp=float(objective(base.detach()+eps*d,q,a,b,'a')[0]);lm=float(objective(base.detach()-eps*d,q,a,b,'a')[0])
   num=(lp-lm)/(2*eps);row[str(eps)]={'numerical_objective_derivative':num,'relative_error':abs(float((g*d).sum())-num)/max(abs(float((g*d).sum())),1e-4)}
  directions.append(row)
 gradient={'loss':float(loss.detach()),'gradient_norm':float(g.norm()),'finite':bool(torch.isfinite(g).all()),'directions':directions,'master_dtype':'float32','decoder_dtype':str(dtype),'loss_dtype':'float32','cast_boundary':'external master -> decoder dtype'}
 def textual_controls(r):
  val=r['value'];other=next(v for v in manifest['values'] if v!=val);q=r['template'].format(entity=r['entity']);unrelated=next(x for x in rows if x['entity']!=r['entity'])['value'];cf=next(v for v in manifest['values'] if v not in [val,unrelated]);out={}
  for name,evidence in [('query_only',None),('relevant_text',val),('unrelated_text',unrelated),('counterfactual_text',cf)]:
   text=q if evidence is None else f'Memory evidence: entity {r["entity"]} has color {evidence}. '+q;out[name]=score(slot_base,text,val,other)
  return out
 textual={str(r['id']):textual_controls(r) for r in rows[8:12]};oracle=[];updates=0
 for i in range(4):
  entity=f'oracle{i}';q=f'What color belongs to entity {entity}?';a=['red','blue','green','yellow'][i];b=['blue','green','yellow','red'][i];pair={'pair':i,'query':q,'a':a,'b':b,'tokenizations':{a:aids(a),b:aids(b)}}
  for which in ['a','b']:
   torch.manual_seed(SEED+100+i*2+(which=='b'));slot=slot_base.clone().requires_grad_(True);opt=torch.optim.SGD([slot],lr=.2);traj=[];initial=score(slot.detach(),q,a,b);zero=score(torch.zeros_like(slot),q,a,b);random=score(torch.randn_like(slot),q,a,b);genuine=score(slot_base,q,a,b)
   for step in range(3):
    opt.zero_grad();obj,nll,margin=objective(slot,q,a,b,which);obj.backward();opt.step();updates+=1;traj.append({'step':step+1,'objective':float(obj.detach()),'nll':nll,'margin':margin,'gradient_norm':float(slot.grad.norm())})
   pair[which]={'initial':initial,'zero':zero,'random':random,'genuine':genuine,'final':score(slot.detach(),q,a,b),'trajectory':traj}
  oracle.append(pair)
 success=sum(1 for p in oracle if p['a']['final']['margin']>0 and p['b']['final']['margin']<0);result={'pairs':oracle,'successful_reversals':success,'pair_count':4,'minimum_required_reversals':4,'passed':success==4,'optimizer_updates':updates,'objective':'answer NLL + 0.5*softplus(-correct margin)','textual_controls':textual,'gradient':gradient,'total_seconds':time.monotonic()-start,'decoder_parameters_frozen':all(not p.requires_grad for p in model.parameters()),'decoder_parameter_grads_absent':all(p.grad is None for p in model.parameters()),'protected_final_holdout_accessed':False};(ROOT/'numerical-oracle-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'updates':updates,'passed':result['passed'],'successful_reversals':success,'gradient':gradient,'seconds':result['total_seconds']}))
if __name__=='__main__':main()
