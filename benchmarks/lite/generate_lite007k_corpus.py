from __future__ import annotations
import json
from pathlib import Path

def build(seed=20261121):
 records=[]; scenarios=[]; queries=[]; families=['installation','configuration','recovery','permissions','storage','network']
 for n,size in enumerate([8,16,32]*30):
  sid=f'k-scenario-{n:03d}'; fam=families[n%len(families)]; eids=[]
  for j in range(size):
   eid=f'{sid}-e{j:03d}'; eids.append(eid); records.append({'access_scope':'public','chunk_id':'fact','collection_id':sid,'content':f'Independent LITE-007K scenario {n:03d} {fam} record {j:03d}: subject k{n:03d} relation r{j:03d} year 2025.', 'evidence_id':eid,'provenance':'LITE-007K independently generated development-evaluation corpus','record_kind':'fact','scenario_id':sid,'source_revision':'lite007k-r1','source_timestamp':'2025-01-01','tags':[fam,'independent']})
  scenarios.append({'collection_size':size,'family':fam,'query_ids':[f'{sid}-q{i}' for i in range(4)],'record_ids':eids,'scenario_id':sid,'split':'development_evaluation'})
  for qi in range(4):
   req=[eids[0]] if qi==3 else [eids[0],eids[qi%3+1]]
   queries.append({'access_scope':'public','expected_abstention':False,'historical_cutoff':'2025-12-31','query':f'For independent subject k{n:03d}, retrieve {fam} evidence case {qi}.','query_id':f'{sid}-q{qi}','required_evidence_ids':req,'scenario_id':sid})
 return {'dataset_version':'lite007k-independent-dev-eval-v1','holdout_included':False,'queries':queries,'query_count':len(queries),'record_count':len(records),'records':records,'scenario_count':len(scenarios),'scenarios':scenarios,'schema_version':1,'seed':seed}
if __name__=='__main__':
 out=Path('benchmarks/lite/fixtures/lite007k-independent'); out.mkdir(parents=True,exist_ok=True); d=build(); (out/'dataset.json').write_text(json.dumps(d,sort_keys=True,indent=2)+'\n'); (out/'manifest.json').write_text(json.dumps({'dataset_version':d['dataset_version'],'seed':d['seed'],'holdout_included':False,'scenario_count':d['scenario_count'],'query_count':d['query_count'],'record_count':d['record_count'],'collection_sizes':[8,16,32],'purpose':'independent development-evaluation only; no final holdout'},sort_keys=True,indent=2)+'\n'); print(d['scenario_count'],d['query_count'],d['record_count'])
