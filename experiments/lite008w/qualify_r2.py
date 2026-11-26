from __future__ import annotations
import hashlib,json,os,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from experiments.lite008p.p_experiment import run_forward,tokid,ids_for,MODEL
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'docs/memorilla-lite/audits/LITE-008W-R1-RUNTIME-REQUALIFICATION.json'
R1=Path.home()/'.cache/memorilla-lite/LITE-008V-R1/original-generation.json'; B=Path.home()/'.cache/memorilla-lite/LITE-008V-R2/reconstruction.pt'; R2=ROOT/'docs/memorilla-lite/audits/LITE-008V-R2-TEACHER-REQUALIFICATION.json'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1);t=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);m=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True);m.requires_grad_(False);m.eval(); bundle=torch.load(B,map_location='cpu'); td=json.loads(R1.read_text()); historical=json.loads(R2.read_text()); rows=json.loads((ROOT/'experiments/lite008p/dataset-manifest.json').read_text())['rows']; by={x['case_id']:x for x in rows}; out=[]
 for i,z in enumerate(td['results']):
  r=by[z['case_id']]; teacher=torch.tensor(z['final_slots'][0],dtype=torch.float32); feat=torch.tensor(json.loads((Path.home()/'.cache/memorilla-lite/LITE-008V-R2/features.json').read_text())['features'][i]['feature'],dtype=torch.float32); w=bundle['W']; eos=bundle['eos']; recon=eos+feat@w.T; a=tokid(t,r['answer_a']);bb=tokid(t,r['answer_b']); ids=ids_for(t,r['query'],1); ot=run_forward(m,ids,teacher.reshape(1,-1),a,bb); orr=run_forward(m,ids,recon.reshape(1,-1),a,bb); exp=historical['records'][i]; out.append({'case_id':z['case_id'],'target':z['target'],'candidate_token_ids':[a,bb],'historical_margin':exp['historical_margin'],'teacher_margin':ot['margin'],'reconstructed_margin':orr['margin'],'teacher_preference':'A' if ot['margin']>0 else 'B','reconstructed_preference':'A' if orr['margin']>0 else 'B','teacher_margin_delta':ot['margin']-exp['historical_margin'],'reconstructed_matches_r2_preference':(('A' if orr['margin']>0 else 'B')==z['target'])})
 result={'status':'PASS' if all(x['teacher_preference']==x['target'] and x['reconstructed_preference']==x['target'] for x in out) else 'NO-GO','decoder_model_path':str(MODEL),'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'r2_bundle_sha256':sha(B),'teacher_source_sha256':sha(R1),'r2_report_sha256':sha(R2),'records':out,'teacher_reversals':sum(x['teacher_preference']==x['target'] for x in out),'reconstructed_reversals':sum(x['reconstructed_preference']==x['target'] for x in out),'optimizer_updates':0,'teacher_regeneration':False,'deterministic_repeat':'not repeated in this bounded qualification; R2 repeat evidence retained'}
 OUT.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps({'status':result['status'],'teacher_reversals':result['teacher_reversals'],'reconstructed_reversals':result['reconstructed_reversals']}))
if __name__=='__main__':main()
