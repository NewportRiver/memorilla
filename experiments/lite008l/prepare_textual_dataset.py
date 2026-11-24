"""Create the LITE-008L textual competence calibration/development manifest."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
from experiments.lite008l.ledger import phase_start,phase_end
from transformers import AutoTokenizer
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008L';SEED=20261124
FORMATS=['question_answer','evidence_question','completion']
VALUES=['red','blue','green','yellow','deep blue','bright red']

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  while c:=f.read(1<<20):h.update(c)
 return h.hexdigest()

def main():
 phase_start('dataset_prepare',seed=SEED)
 ROOT.mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True)
 rows=[]
 familiar=[('France','Paris'),('Germany','Berlin'),('Italy','Rome'),('Japan','Tokyo')]
 for i,(e,v) in enumerate(familiar): rows.append({'id':f'f{i:02d}','kind':'familiar','entity':e,'value':v,'alternate':familiar[(i+1)%len(familiar)][1],'format':FORMATS[i%3]})
 entities=['cobalt','linen','marble','cedar','orbit','quartz','velvet','harbor']
 for i,e in enumerate(entities):
  v=VALUES[(i*3+1)%len(VALUES)];alt=VALUES[(i*3+2)%len(VALUES)];rows.append({'id':f'r{i:02d}','kind':'randomized','entity':e,'value':v,'alternate':alt,'format':FORMATS[i%3]})
 splits={'calibration':[r['id'] for r in rows[:4]],'development':[r['id'] for r in rows[4:]]}
 tokenizations={};
 for r in rows:
  for v in [r['value'],r['alternate']]:
   tokenizations.setdefault(v,tok(' '+v,add_special_tokens=False)['input_ids'])
  assert r['value'].lower() not in r['entity'].lower()
 manifest={'schema':'LITE-008L-textual-dataset-v1','seed':SEED,'rows':rows,'partitions':splits,'prompt_formats':FORMATS,'answer_in_query':False,'tokenizations':tokenizations,'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'model_config_sha256':sha(MODEL/'config.json'),'protected_final_holdout_accessed':False}
 (ROOT/'textual-dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');phase_end('dataset_prepare',rows=len(rows),calibration=len(splits['calibration']),development=len(splits['development']),manifest_sha256=sha(ROOT/'textual-dataset-manifest.json'))
 print(json.dumps({'rows':len(rows),'calibration':len(splits['calibration']),'development':len(splits['development']),'manifest_sha256':sha(ROOT/'textual-dataset-manifest.json')}))
if __name__=='__main__':main()
