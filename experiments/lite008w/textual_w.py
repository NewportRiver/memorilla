import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path(__file__).resolve().parents[2];AUD=ROOT/'docs/memorilla-lite/audits'
def score(m,t,text,a,b):
 ids=torch.tensor([t(text,add_special_tokens=True)['input_ids']]); log=m(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False).logits.float()[:,-1,:]; ia=t(' '+a,add_special_tokens=False)['input_ids'];ib=t(' '+b,add_special_tokens=False)['input_ids'];
 if len(ia)!=1 or len(ib)!=1: raise ValueError('candidate is not single token')
 return {'margin':float(log[0,ia[0]]-log[0,ib[0]]),'a_token_id':ia[0],'b_token_id':ib[0],'a_logit':float(log[0,ia[0]]),'b_logit':float(log[0,ib[0]])}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);d=json.loads((AUD/'LITE-008W-R2-DATASET-MANIFEST.json').read_text());t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval(); rows=[];st=time.monotonic()
 for r in d['rows']:
  q=r['query'];base=score(m,t,q+' Answer:',r['answer_a'],r['answer_b']);rel=score(m,t,q+' '+r['evidence_a']+' Answer:',r['answer_a'],r['answer_b']);cf=score(m,t,q+' '+r['evidence_b']+' Answer:',r['answer_a'],r['answer_b']);un=score(m,t,q+' '+r['unrelated_evidence']+' Answer:',r['answer_a'],r['answer_b']);ok=abs(base['margin'])<=4 and rel['margin']>=1 and cf['margin']<=-1 and un['margin']>-1;rows.append({'id':r['id'],'entity':r['entity'],'partition':r['partition'],'query_only':base,'relevant':rel,'counterfactual':cf,'unrelated':un,'eligible':ok,'rejection_reasons':[] if ok else ['predeclared textual rule not satisfied']})
 out={'schema':'LITE-008W-R2-textual-competence-v1','status':'COMPLETED','rows':rows,'eligible_count':sum(x['eligible'] for x in rows),'eligible_ids':[x['id'] for x in rows if x['eligible']],'forward_count':len(rows)*4,'latent_outcomes_used_for_selection':False,'protected_final_holdout_accessed':False,'optimizer_updates':0,'wall_seconds':time.monotonic()-st};(AUD/'LITE-008W-R3-TEXTUAL-COMPETENCE-RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'eligible':out['eligible_count'],'total':len(rows),'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
