"""Forward-only progressive compression of textual evidence embeddings."""
import json,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import MODEL,ROOT,ids_for,tokid,run_forward

def main():
 torch.set_num_threads(1);d=json.loads((ROOT/'dataset-manifest.json').read_text());r=next(x for x in d['rows'] if x['case_id']=='case_00');t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();a=tokid(t,r['answer_a']);b=tokid(t,r['answer_b']);ev=t(r['relevant_evidence'],add_special_tokens=False)['input_ids'];e=m.get_input_embeddings();v=e(torch.tensor([ev])).detach().float()[0];out={}
 for n in [1,2,4]:
  chunks=torch.chunk(v,n,dim=0);slots=torch.stack([x.mean(0) for x in chunks]);ids=ids_for(t,r['query'],n);out[str(n)]=run_forward(m,ids,slots,a,b)
 (ROOT/'compressed-text-controls.json').write_text(json.dumps({'schema':'LITE-008P-compressed-text-controls-v1','case_id':r['case_id'],'evidence_token_count':len(ev),'results':out,'interpretation':'compressed representations are diagnostics, not independent learned memory'},indent=2)+'\n');print(json.dumps({k:v['margin'] for k,v in out.items()}))
if __name__=='__main__':main()
