"""Forward-only final embedding and greedy next-token evaluation."""
from __future__ import annotations
import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008o.optimize_oracle import make_ids,token,forward,MODEL
ROOT=Path(__file__).resolve().parent;CACHE=Path.home()/'.cache/memorilla-lite/LITE-008O'
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();d=json.loads((CACHE/'optimization-contrastive-results.json').read_text());rows=[]
 for r in d['results']:
  ids=make_ids(tok,next(x for x in json.loads((ROOT/'dataset-manifest.json').read_text())['rows'] if x['case_id']==r['case_id'])['query']);a=token(tok,next(x for x in json.loads((ROOT/'dataset-manifest.json').read_text())['rows'] if x['case_id']==r['case_id'])['answer_a']);b=token(tok,next(x for x in json.loads((ROOT/'dataset-manifest.json').read_text())['rows'] if x['case_id']==r['case_id'])['answer_b']);slot=torch.tensor(r['final_embedding']);ev=forward(m,ids,slot,a,b);rows.append({'case_id':r['case_id'],'mode':r['mode'],'target_direction':r['target_direction'],'a_token':tok.decode([a]),'b_token':tok.decode([b]),'greedy_next_token':tok.decode([ev['greedy_id']]),'greedy_next_token_id':ev['greedy_id'],'answer_nll':{'a':ev['nll_a'],'b':ev['nll_b']},'margin_a_minus_b':ev['margin_a_minus_b'],'forward_count':ev['complete_decoder_forwards'],'backward_vjps':ev['backward_vjps']})
 out={'schema':'LITE-008O-forward-only-final-evaluation-v1','status':'completed','rows':rows,'forward_count':len(rows),'backward_vjps':0,'wall_seconds':time.monotonic()-st,'protected_final_holdout_accessed':False};(CACHE/'forward-only-final-evaluation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'rows':len(rows),'seconds':out['wall_seconds']}))
if __name__=='__main__':main()
