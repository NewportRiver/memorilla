from __future__ import annotations
import hashlib, json, random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/lite/fixtures/lite007i'
SEED=20261120
SIZES=(8,16,32,64,128)
FAMILIES=('installation','resource','dependency','environment','recovery','supersession','combination','temporal','cross_document','hard_negative')

def sha(path):
 h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()

def build():
 rng=random.Random(SEED); scenarios=[]; records=[]; queries=[]
 for i in range(300):
  sid=f'i-scenario-{i:03d}'; size=SIZES[i%len(SIZES)]; split='train' if size<=32 else 'validation'; family=FAMILIES[i%len(FAMILIES)]; scope='public'; base=2024+(i%3)
  ids=[]
  for j in range(size):
   eid=f'{sid}-e{j:03d}'; ids.append(eid)
   role=('fact' if j==0 else 'support' if j==1 else 'correction' if j==2 else 'hard_negative' if j%7==0 else 'context')
   text=f'Independent scenario {i:03d} {family} record {j:03d}: subject token {i:03d} has role {role}; event year {base}; relation marker {i:03d}-{j:03d}.'
   if role=='correction': text += f' This supersedes the earlier state for scenario {i:03d}.'
   if role=='hard_negative': text += f' Similar vocabulary is unrelated to the required relation for scenario {i:03d}.'
   records.append({'evidence_id':eid,'content':text,'provenance':'LITE-007I synthetic development corpus','source_revision':'lite007i-r1','access_scope':scope,'source_timestamp':f'{base}-01-{(j%27)+1:02d}','collection_id':sid,'chunk_id':role,'record_kind':role,'tags':[family,role], 'scenario_id':sid})
  queries += [
   {'query_id':f'{sid}-q0','scenario_id':sid,'query':f'Which separate records establish the fact and its correction for scenario {i:03d}?','required_evidence_ids':[ids[0],ids[2]],'expected_abstention':False,'access_scope':scope,'historical_cutoff':f'{base}-12-31','category':'multi_evidence'},
   {'query_id':f'{sid}-q1','scenario_id':sid,'query':f'Which records connect the fact to supporting context for scenario {i:03d}?','required_evidence_ids':[ids[0],ids[1]],'expected_abstention':False,'access_scope':scope,'historical_cutoff':f'{base}-12-31','category':'cross_record'},
   {'query_id':f'{sid}-q2','scenario_id':sid,'query':f'Which record proves an unavailable event for scenario {i:03d}?','required_evidence_ids':[],'expected_abstention':True,'access_scope':scope,'historical_cutoff':f'{base}-12-31','category':'insufficient_evidence'},
   {'query_id':f'{sid}-q3','scenario_id':sid,'query':f'Which record is the unrelated hard negative for scenario {i:03d}?','required_evidence_ids':[next(x for x in ids if x.endswith('000'))],'expected_abstention':False,'access_scope':scope,'historical_cutoff':f'{base}-12-31','category':'hard_negative'}
  ]
  scenarios.append({'scenario_id':sid,'split':split,'family':family,'collection_size':size,'record_ids':ids,'query_ids':[f'{sid}-q{q}' for q in range(4)]})
 payload={'schema_version':1,'dataset_version':'lite007i-dev-v1','seed':SEED,'scenario_count':300,'record_count':len(records),'query_count':len(queries),'holdout_included':False,'scenarios':scenarios,'records':records,'queries':queries}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'dataset.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
 manifest={'schema_version':1,'dataset_version':payload['dataset_version'],'seed':SEED,'counts':{'scenarios':300,'records':len(records),'queries':len(queries),'train_scenarios':180,'validation_scenarios':120,'train_queries':720,'validation_queries':480},'collection_sizes':{str(x):sum(s['collection_size']==x for s in scenarios) for x in SIZES},'dataset_sha256':sha(OUT/'dataset.json'),'holdout_included':False,'scenario_disjoint':True}
 (OUT/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': build()
