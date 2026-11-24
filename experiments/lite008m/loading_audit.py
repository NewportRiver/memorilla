"""Audit supported Transformers loading modes without forward/backward execution."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from experiments.lite008l.ledger import record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008M'
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--low-memory',action='store_true');ap.add_argument('--dtype',choices=['bfloat16','float32'],default='float32');a=ap.parse_args();dtype=torch.bfloat16 if a.dtype=='bfloat16' else torch.float32;mode=f'{a.dtype}_low_cpu_mem_{a.low_memory}';start=time.monotonic();record('loading_audit','started',mode=mode,low_cpu_mem_usage=a.low_memory,dtype=a.dtype)
 try:
  m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=dtype,low_cpu_mem_usage=a.low_memory);m.eval();sd=m.state_dict();
  keys=['model.embed_tokens.weight','model.layers.0.self_attn.q_proj.weight','lm_head.weight'];checks={};
  with safe_open(str(MODEL/'model.safetensors'),framework='pt',device='cpu') as f:
   for k in keys:
    if k in sd and k in f.keys():
     src=f.get_tensor(k).float();loaded=sd[k].detach().cpu().float();checks[k]={'shape_equal':list(src.shape)==list(loaded.shape),'max_abs_diff':float((src-loaded).abs().max()),'mean_abs_diff':float((src-loaded).abs().mean())}
  tied=m.get_input_embeddings().weight.data_ptr()==m.get_output_embeddings().weight.data_ptr();res={'schema':'LITE-008M-loading-audit-v1','mode':mode,'status':'completed','low_cpu_mem_usage':a.low_memory,'dtype':a.dtype,'parameter_count':sum(p.numel() for p in m.parameters()),'checks':checks,'tied_input_output_weights':tied,'config_model_type':m.config.model_type,'loading_seconds':time.monotonic()-start};(ROOT/f'loading-{a.dtype}-lowmem-{a.low_memory}.json').write_text(json.dumps(res,indent=2)+'\n');record('loading_audit','completed',mode=mode,loading_seconds=res['loading_seconds'],tied_weights=tied);print(json.dumps(res))
 except Exception as e:
  record('loading_audit','aborted',mode=mode,error=type(e).__name__+': '+str(e));print(json.dumps({'status':'aborted','mode':mode,'error':type(e).__name__+': '+str(e)}));raise
if __name__=='__main__':main()
