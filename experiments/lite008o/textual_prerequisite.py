"""Forward-only ordinary textual prerequisite for LITE-008O."""
from __future__ import annotations
import json,math,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path(__file__).resolve().parent

def score(m,tok,text,a,b):
 ids=torch.tensor([tok(text,add_special_tokens=True)['input_ids']]);out=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float();last=out[:,-1,:];ia=tok(' '+a,add_special_tokens=False)['input_ids'];ib=tok(' '+b,add_special_tokens=False)['input_ids'];
 if len(ia)!=1 or len(ib)!=1:raise ValueError('non-single-token candidate')
 la=float(last[0,ia[0]]);lb=float(last[0,ib[0]]);return {'a_logit':la,'b_logit':lb,'margin_a_minus_b':la-lb,'forward_count':1}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);start=time.monotonic();data=json.loads((ROOT/'dataset-manifest.json').read_text());tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();rows=[]
 for r in data['rows']:
  base=score(m,tok,r['query']+' Answer:',r['answer_a'],r['answer_b']);rel=score(m,tok,r['query']+' '+r['relevant_evidence']+' Answer:',r['answer_a'],r['answer_b']);cf=score(m,tok,r['query']+' '+r['counterfactual_evidence']+' Answer:',r['answer_a'],r['answer_b']);un=score(m,tok,r['query']+' '+r['unrelated_evidence']+' Answer:',r['answer_a'],r['answer_b']);eligible=(abs(base['margin_a_minus_b'])<=4.0 and rel['margin_a_minus_b']>=1.0 and cf['margin_a_minus_b']<=-1.0 and un['margin_a_minus_b']>-1.0);rows.append({'case_id':r['case_id'],'partition':r['partition'],'query_only':base,'relevant':rel,'counterfactual':cf,'unrelated':un,'eligible':eligible,'rejection_reasons':[] if eligible else ['fixed textual eligibility rule not satisfied']})
 oracle=[x for x in rows if x['partition']=='oracle'];eligible=[x['case_id'] for x in oracle if x['eligible']];out={'schema':'LITE-008O-textual-prerequisite-v1','status':'completed','eligibility_rule':'predeclared manifest rule; no latent optimization outcomes used','rows':rows,'eligible_oracle_cases':eligible,'eligible_count':len(eligible),'required_oracle_pairs':4,'forward_count':len(rows)*4,'wall_seconds':time.monotonic()-start,'protected_final_holdout_accessed':False};(ROOT/'textual-prerequisite-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'eligible':eligible,'eligible_count':len(eligible),'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
