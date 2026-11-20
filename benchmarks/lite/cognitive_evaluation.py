from __future__ import annotations
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent/'fixtures'
def load():
 s=json.loads((ROOT/'development_sources.json').read_text()); q=json.loads((ROOT/'development_queries.json').read_text()); return s,q
def validate():
 s,q=load(); sources=s['sources']; queries=q['queries']; assert len({x['scenario_id'] for x in queries})==12; assert len(queries)>=120; assert len({x['query_id'] for x in queries})==len(queries); ids={x['evidence_id'] for x in sources}; assert len(ids)==len(sources)
 for x in queries:
  assert x['scenario_id'] and x['category'] and x['access_scope']; assert set(x['required_evidence_ids']).issubset(ids); assert set(x['known_distractors']).issubset(ids); assert x['scenario_id'] in x['query_id']
 canonical=json.dumps(queries,sort_keys=True,separators=(',',':')).encode(); return {'schema_version':1,'sources':len(sources),'queries':len(queries),'scenario_count':len({x['scenario_id'] for x in queries}),'query_hash':hashlib.sha256(canonical).hexdigest(),'categories':{c:sum(x['category']==c for x in queries) for c in sorted({x['category'] for x in queries})},'holdout_labels':'not present; external sealed partition required'}
if __name__=='__main__': print(json.dumps(validate(),sort_keys=True,indent=2))
