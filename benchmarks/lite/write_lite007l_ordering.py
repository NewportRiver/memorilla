import json,re
from pathlib import Path
d=json.loads(Path('benchmarks/lite/fixtures/lite007k-independent/dataset.json').read_text()); groups={}
def norm(s): return re.sub(r'\d+','<N>',re.sub(r'\s+',' ',s.lower()).strip())
for r in d['records']: groups.setdefault(norm(r['content']),[]).append(r['evidence_id'])
rows=[]
for k,v in groups.items():
 if len(v)>1:
  sids={next(s['scenario_id'] for s in d['scenarios'] if e in s['record_ids']) for e in v}; splits={next(s['split'] for s in d['scenarios'] if e in s['record_ids']) for e in v}; rows.append({'template':k,'evidence_ids':v,'scenario_ids':sorted(sids),'partitions':sorted(splits),'cross_scenario':len(sids)>1})
Path('docs/memorilla-lite/audits/LITE-007L-ORDERING-CONTROLS.json').write_text(json.dumps({'evidence_order_policy':'preserve dataset order; stable mergesort ties; no label-based candidate ordering','normalized_collision_groups':rows,'final_evaluation_permutation_required':True,'tie_breaking':'stable original candidate order','zero_direct_initialization_is_tie_sensitive':True},sort_keys=True,indent=2)+'\n')
