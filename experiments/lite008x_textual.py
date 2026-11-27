import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import MODEL
ROOT=Path(__file__).resolve().parents[1];AUD=ROOT/'docs/memorilla-lite/audits';MAN=AUD/'LITE-008X-DATASET-MANIFEST.json'
def score(m,t,text,a,b):
 ids=t(text,add_special_tokens=True)['input_ids']; ia=t(' '+a,add_special_tokens=False)['input_ids'];ib=t(' '+b,add_special_tokens=False)['input_ids']
 if len(ia)!=1 or len(ib)!=1: raise ValueError((a,ia,b,ib))
 inp=t(text+' Answer:',return_tensors='pt');
 if inp['input_ids'].shape[1]>32: raise RuntimeError(('sequence',inp['input_ids'].shape[1],text))
 with torch.no_grad(): log=m(**inp,use_cache=False).logits.float()[:,-1,:]
 return {'margin':float(log[0,ia[0]]-log[0,ib[0]]),'a_token_id':ia[0],'b_token_id':ib[0],'a_logit':float(log[0,ia[0]]),'b_logit':float(log[0,ib[0]]),'greedy_id':int(log[0].argmax())}
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);d=json.loads(MAN.read_text());t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();st=time.monotonic();rows=[]
 for r in d['rows']:
  a,b=r['answer_a'],r['answer_b']; q=r['query']; rel=r['evidence_a'];cf=r['evidence_b']; wrong=f'Record unrelated stores value {a}.'; conflict=f'{r["entity"]} stores {a} and {b}.'; variants={'query_only':q,'relevant':q+' '+rel,'counterfactual':q+' '+cf,'unrelated':q+' '+r['unrelated_evidence'],'same_answer_wrong_entity':q+' '+wrong,'conflicting':q+' '+conflict}
  scores={k:score(m,t,v,a,b) for k,v in variants.items()}; eligible=abs(scores['query_only']['margin'])<=4 and scores['relevant']['margin']>=1 and scores['counterfactual']['margin']<=-1 and scores['unrelated']['margin']>-1
  rows.append({'id':r['id'],'group':r['group'],'textual_eligible':eligible,'rejection_reasons':[] if eligible else ['predeclared composite textual criterion failed'],'scores':scores})
 out={'schema':'LITE-008X-textual-v1','status':'COMPLETED','rows':rows,'eligible_by_group':{g:sum(x['textual_eligible'] for x in rows if x['group']==g) for g in 'ABCD'},'forward_count':len(rows)*6,'optimizer_updates':0,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False};(AUD/'LITE-008X-TEXTUAL-COMPETENCE-RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['eligible_by_group']))
if __name__=='__main__':main()
