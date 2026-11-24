"""One-update LITE-008O computational preflight."""
import json,time,torch
from pathlib import Path
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008o.optimize_oracle import make_ids,token,objective,forward,MODEL
ROOT=Path(__file__).resolve().parent
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);st=time.monotonic();d=json.loads((ROOT/'dataset-manifest.json').read_text())['rows'][8];tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval();ids=make_ids(tok,d['query']);a=token(tok,d['answer_a']);b=token(tok,d['answer_b']);slot=m.get_input_embeddings()(ids).detach().float()[0,0];x=objective(m,ids,slot,a,b,True,True);f=forward(m,ids,slot,a,b);out={'schema':'LITE-008O-preflight-v1','objective_seconds':time.monotonic()-st,'gradient_norm':float(x['grad'].norm()),'objective':x['loss'],'forward_margin':f['margin_a_minus_b'],'expected_updates':96,'resource_policy':'existing 1GiB RSS/536MiB reserve/300s subprocess/30min cumulative','protected_final_holdout_accessed':False};(ROOT/'preflight-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
