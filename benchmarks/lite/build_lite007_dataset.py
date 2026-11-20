"""Build the versioned, scenario-disjoint LITE-007 synthetic development corpus."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'benchmarks/lite/fixtures/lite007'
SEED=20261120

families=[
 ('installation','package resolver','ABI mismatch','repository metadata','dependency lock'),
 ('resource','worker pool','memory ceiling','queue pressure','bounded batch'),
 ('dependency','runtime plugin','incompatible symbol','version pin','compatibility check'),
 ('environment','Android process','permission boundary','scope configuration','restart probe'),
 ('assumption','cache path','stale assumption','fresh inspection','reproduction run'),
 ('tool','artifact scanner','timeout','smaller chunk','repeatable scan'),
 ('recovery','database migration','partial write','transaction rollback','integrity check'),
 ('recommendation','deployment plan','conflicting advice','source revision','review trace'),
 ('supersession','encoder artifact','obsolete revision','new manifest','checksum comparison'),
 ('combination','attention pipeline','shape mismatch','contract adapter','cross-module test'),
]
verbs=['observed','recorded','isolated','reproduced','measured','confirmed']

def h(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rec(eid, text, scope, ts, sid, kind, tags):
 return {'evidence_id':eid,'content':text,'provenance':'LITE-007 synthetic development corpus','source_revision':'lite007-r1','access_scope':scope,'source_timestamp':ts,'collection_id':sid,'chunk_id':kind,'chunk_start':0,'chunk_end':len(text),'record_kind':kind,'tags':tags}

def build():
 rng=random.Random(SEED); scenarios=[]; records=[]; queries=[]
 scenario_ids=[f'scenario-{i:03d}' for i in range(300)]
 rng.shuffle(scenario_ids)
 train=set(scenario_ids[:240]); validation=set(scenario_ids[240:])
 for i,sid in enumerate(sorted(scenario_ids)):
  fam,subject,problem,repair,check=families[i%len(families)]
  scope='private' if i%30==0 else 'public'
  base=20250101+(i%27)*100+(i%9)
  dates=[f'{base//10000:04d}-{(base//100)%100:02d}-{base%100:02d}',f'2025-{(i%9)+1:02d}-12',f'2025-{(i%9)+1:02d}-18',f'2025-{(i%9)+1:02d}-24',f'2025-{(i%9)+1:02d}-28']
  nonce=f'{rng.choice(verbs)} {rng.choice(["during", "after", "before", "within"])} run {i:03d}'
  ids={k:f'{sid}-{k}' for k in ('observation','cause','repair','validation','distractor')}
  records += [
   rec(ids['observation'],f'Run {i:03d} {nonce}: the {subject} reported {problem} while processing the bounded workload.',scope,dates[0],sid,'observation',[fam,'failure','historical']),
   rec(ids['cause'],f'Investigation for run {i:03d} traced the {subject} failure to {problem}; the evidence was reproduced with {rng.choice(["a clean process", "a fresh workspace", "a replayed input"])}.',scope,dates[1],sid,'cause',[fam,'cause','historical']),
   rec(ids['repair'],f'The corrective action for run {i:03d} applied the {repair} and removed the incompatible condition without changing authoritative source data.',scope,dates[2],sid,'repair',[fam,'correction','historical']),
   rec(ids['validation'],f'After the correction in run {i:03d}, the {check} completed successfully and the recorded artifact checksum remained stable.',scope,dates[3],sid,'validation',[fam,'success','historical']),
   rec(ids['distractor'],f'A neighboring run {i:03d} used the same {subject} vocabulary but investigated a different {rng.choice(["logging delay", "network retry", "display warning", "unrelated cache"])}; it is not evidence for the primary failure.',scope,dates[4],sid,'distractor',[fam,'similar','irrelevant']),
  ]
  qscope=scope
  queries += [
   {'query_id':f'{sid}-q0','scenario_id':sid,'query':f'Which historical records connect the {problem} diagnosis to the {repair} correction?','required_evidence_ids':[ids['cause'],ids['repair']],'known_distractors':[ids['distractor']],'expected_abstention':False,'policy_expected_abstention':False,'access_scope':qscope,'historical_cutoff':dates[3],'category':'multi_hop'},
   {'query_id':f'{sid}-q1','scenario_id':sid,'query':f'What observation and later verification belong to the same bounded {fam} run?','required_evidence_ids':[ids['observation'],ids['validation']],'known_distractors':[ids['distractor']],'expected_abstention':False,'policy_expected_abstention':False,'access_scope':qscope,'historical_cutoff':dates[4],'category':'multi_evidence'},
   {'query_id':f'{sid}-q2','scenario_id':sid,'query':f'Which record proves the unrecorded battery discharge for run {i:03d}?','required_evidence_ids':[],'known_distractors':[ids['observation'],ids['distractor']],'expected_abstention':True,'policy_expected_abstention':False,'access_scope':qscope,'historical_cutoff':dates[4],'category':'insufficient_evidence'},
   {'query_id':f'{sid}-q3','scenario_id':sid,'query':f'Which historical record concerns the neighboring but irrelevant event for run {i:03d}?','required_evidence_ids':[ids['distractor']],'known_distractors':[],'expected_abstention':False if scope=='public' else True,'policy_expected_abstention':False if scope=='public' else True,'access_scope':'public' if scope=='private' else scope,'historical_cutoff':dates[4],'category':'access_policy' if scope=='private' else 'irrelevant_similarity'},
  ]
  scenarios.append({'scenario_id':sid,'split':'train' if sid in train else 'validation','family':fam,'scope':scope,'record_ids':list(ids.values()),'query_ids':[f'{sid}-q{i}' for i in range(4)]})
 payload={'schema_version':1,'dataset_version':'lite007-dev-v1','seed':SEED,'scenario_count':300,'record_count':len(records),'query_count':len(queries),'train_scenarios':240,'validation_scenarios':60,'scenarios':scenarios,'records':records,'queries':queries,'holdout_included':False}
 OUT.mkdir(parents=True,exist_ok=True)
 (OUT/'dataset.json').write_text(json.dumps(payload,sort_keys=True,indent=2)+'\n')
 manifest={'schema_version':1,'dataset_version':'lite007-dev-v1','seed':SEED,'files':{'dataset.json':h(OUT/'dataset.json')},'counts':{'scenarios':300,'records':len(records),'queries':len(queries),'train_scenarios':240,'validation_scenarios':60,'train_records':1200,'validation_records':300,'train_queries':960,'validation_queries':240},'split_rule':'scenario-disjoint deterministic shuffle by seed; no query-level random split','holdout_labels_included':False,'source_policy':'synthetic only; no sealed labels; no private mission data'}
 (OUT/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 print(json.dumps(manifest,sort_keys=True,indent=2))
if __name__=='__main__': build()
