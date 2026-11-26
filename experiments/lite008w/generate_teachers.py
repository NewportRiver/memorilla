import json,time,hashlib
from pathlib import Path
import torch
from experiments.lite008p.p_experiment import train
from lite.encoder.onnx_encoder import MiniLMEncoder
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2]; AUD=ROOT/'docs/memorilla-lite/audits'; CACHE=Path.home()/'.cache/memorilla-lite/LITE-008W-R2'; CACHE.mkdir(parents=True,exist_ok=True); MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(1);torch.set_num_interop_threads(1); started=time.monotonic(); manifest=json.loads((AUD/'LITE-008W-R2-DATASET-MANIFEST.json').read_text()); manifest_hash=sha(AUD/'LITE-008W-R2-DATASET-MANIFEST.json'); tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True); model=AutoModelForCausalLM.from_pretrained(str(MODEL),local_files_only=True,use_safetensors=True,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True); model.requires_grad_(False);model.eval(); enc=MiniLMEncoder(threads=1); features=[]
 for r in manifest['rows']:
  for target,evidence in [('A',r['evidence_a']),('B',r['evidence_b'])]:
   v=enc.encode(evidence).astype('float32'); features.append({'id':r['id'],'entity':r['entity'],'partition':r['partition'],'target':target,'query':r['query'],'evidence':evidence,'feature':v.tolist(),'feature_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'manifest_sha256':manifest_hash,'encoder_identity':enc.identity()})
 enc.close(); (CACHE/'features.json').write_text(json.dumps({'manifest_sha256':manifest_hash,'features':features},indent=2)+'\n'); ledger=CACHE/'update-ledger.jsonl'; results=[]; updates=0
 for r in manifest['rows']:
  rr=dict(r); rr['case_id']=r['id']
  for target in (True,False):
   # target direction uses the matching factual answer; identical paired EOS initialization is inside train().
   results.append(train(model,tok,rr,1,target,16,.05,'eos','contrastive',ledger,'lite008w_r2')); updates+=16
 out={'schema':'LITE-008W-R2-teacher-generation-v1','status':'COMPLETED','manifest_sha256':manifest_hash,'results':results,'optimizer_updates':updates,'configured_updates':192,'max_authorized_updates':256,'model_path':str(MODEL),'model_config_hash':sha(MODEL/'config.json'),'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'feature_artifact':str(CACHE/'features.json'),'feature_artifact_sha256':sha(CACHE/'features.json'),'wall_seconds':time.monotonic()-started,'protected_final_holdout_accessed':False}; (CACHE/'teachers.json').write_text(json.dumps(out,indent=2)+'\n'); (AUD/'LITE-008W-R2-TEACHER-GENERATION-RESULTS.json').write_text(json.dumps({'status':'COMPLETED','optimizer_updates':updates,'configured_updates':192,'max_authorized_updates':256,'wall_seconds':out['wall_seconds'],'outside_git_artifact':str(CACHE/'teachers.json'),'manifest_sha256':manifest_hash,'teacher_count':len(results),'failures':[]},indent=2)+'\n'); print(json.dumps({'updates':updates,'seconds':out['wall_seconds'],'artifact':str(CACHE/'teachers.json')}))
if __name__=='__main__':main()
