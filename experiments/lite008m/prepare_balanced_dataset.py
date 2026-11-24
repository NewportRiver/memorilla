"""Create corrected balanced LITE-008M textual data; historical LITE-008L is untouched."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
from transformers import AutoTokenizer
from experiments.lite008l.ledger import record
MODEL=Path.home()/'.cache/memorilla-lite/LITE-008D-R3/model';ROOT=Path.home()/'.cache/memorilla-lite/LITE-008M';SEED=20261124
SINGLE=['red','blue','green','yellow','orange','purple','black','white'];MULTI=['deep blue','bright red','light green','dark yellow']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(1<<20):h.update(c)
 return h.hexdigest()
def main():
 rng=random.Random(SEED);ROOT.mkdir(parents=True,exist_ok=True);record('dataset_prepare','started',seed=SEED,version='balanced-v1')
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True,use_fast=True);rows=[]
 # Separate identities and balanced answer frequencies. Each value is correct three times in development.
 values=(SINGLE*3);rng.shuffle(values);alts=[]
 for i,v in enumerate(values): alts.append(values[(i+len(values)//2)%len(values)])
 for i,(v,a) in enumerate(zip(values,alts)):
  rows.append({'id':f'd{i:02d}','partition':'development','entity':f'entity_{i:02d}_{rng.randrange(100000,999999)}','relation':'color','value':v,'alternate':a,'candidate_order':['A','B'] if i%2==0 else ['B','A'],'answer_orientation':'A' if i%2==0 else 'B','multi_token':False})
 # Four independent multi-token pairs; marked separate and excluded from primary single-token denominator.
 for j,(v,a) in enumerate([('deep blue','bright red'),('bright red','deep blue'),('light green','dark yellow'),('dark yellow','light green')]):
  i=24+j;rows.append({'id':f'm{i:02d}','partition':'development','entity':f'entity_{i:02d}_{rng.randrange(100000,999999)}','relation':'color','value':v,'alternate':a,'candidate_order':['A','B'] if j%2==0 else ['B','A'],'answer_orientation':'A' if j%2==0 else 'B','multi_token':True})
 # Calibration uses disjoint entities and balanced values, not used for prompt selection in M.
 for i,v in enumerate(['red','green','blue','yellow','orange','purple','black','white']):
  a=['blue','red','yellow','green','purple','orange','white','black'][i]
  rows.append({'id':f'c{i:02d}','partition':'calibration','entity':f'calibration_{i:02d}_{rng.randrange(100000,999999)}','relation':'color','value':v,'alternate':a,'candidate_order':['A','B'] if i%2==0 else ['B','A'],'answer_orientation':'A' if i%2==0 else 'B','multi_token':False})
 for r in rows:
  for v in (r['value'],r['alternate']): r.setdefault('tokenizations',{})[v]=tok(' '+v,add_special_tokens=False)['input_ids']
 manifest={'schema':'LITE-008M-balanced-textual-dataset-v1','seed':SEED,'rows':rows,'partitions':{'calibration':[r['id'] for r in rows if r['partition']=='calibration'],'development':[r['id'] for r in rows if r['partition']=='development']},'primary_development_ids':[r['id'] for r in rows if r['partition']=='development' and not r['multi_token']],'multi_token_development_ids':[r['id'] for r in rows if r['multi_token']],'tokenizer_sha256':sha(MODEL/'tokenizer.json'),'model_config_sha256':sha(MODEL/'config.json'),'protected_final_holdout_accessed':False}
 (ROOT/'balanced-dataset-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');freq={v:sum(r['value']==v for r in rows if r['partition']=='development') for v in sorted(set(r['value'] for r in rows if r['partition']=='development'))};record('dataset_prepare','completed',rows=len(rows),development=28,calibration=8,answer_frequency=freq,manifest_sha256=sha(ROOT/'balanced-dataset-manifest.json'));print(json.dumps({'rows':len(rows),'development':28,'calibration':8,'answer_frequency':freq,'manifest_sha256':sha(ROOT/'balanced-dataset-manifest.json')}))
if __name__=='__main__':main()
