"""Forward-only textual prerequisite for fresh P entities."""
import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path(__file__).resolve().parent
def score(m,t,text,a,b):
 ids=torch.tensor([t(text,add_special_tokens=True)['input_ids']]);log=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float()[:,-1,:];ia=t(' '+a,add_special_tokens=False)['input_ids'];ib=t(' '+b,add_special_tokens=False)['input_ids'];return {'margin':float(log[0,ia[0]]-log[0,ib[0]]),'a_logit':float(log[0,ia[0]]),'b_logit':float(log[0,ib[0]])}
def main():
 torch.set_num_threads(1);st=time.monotonic();d=json.loads((ROOT/'dataset-manifest.json').read_text());t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();rows=[]
 for r in d['rows']:
  q=r['query'];base=score(m,t,q+' Answer:',r['answer_a'],r['answer_b']);rel=score(m,t,q+' '+r['relevant_evidence']+' Answer:',r['answer_a'],r['answer_b']);cf=score(m,t,q+' '+r['counterfactual_evidence']+' Answer:',r['answer_a'],r['answer_b']);un=score(m,t,q+' '+r['unrelated_evidence']+' Answer:',r['answer_a'],r['answer_b']);ok=abs(base['margin'])<=4 and rel['margin']>=1 and cf['margin']<=-1 and un['margin']>-1;rows.append({'case_id':r['case_id'],'partition':r['partition'],'query_only':base,'relevant':rel,'counterfactual':cf,'unrelated':un,'eligible':ok,'rejection_reasons':[] if ok else ['fixed textual rule not satisfied']})
 out={'schema':'LITE-008P-textual-prerequisite-v1','rows':rows,'eligible_oracle_cases':[x['case_id'] for x in rows if x['partition']=='oracle' and x['eligible']],'eligible_calibration_cases':[x['case_id'] for x in rows if x['partition']=='calibration' and x['eligible']],'forward_count':len(rows)*4,'wall_seconds':time.monotonic()-st,'latent_outcomes_used_for_selection':False,'protected_final_holdout_accessed':False};(ROOT/'textual-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'eligible_oracle':out['eligible_oracle_cases'],'eligible_calibration':out['eligible_calibration_cases'],'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
