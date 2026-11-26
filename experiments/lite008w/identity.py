"""Fail-closed content-bound identity contract for W artifacts."""
from __future__ import annotations
import hashlib,json

def canonical_identity(record: dict) -> str:
 required=['manifest_sha256','partition','entity','case_id','target','evidence','query','candidate_answers','candidate_token_ids','factual_direction','encoder_identity','decoder_identity','teacher_generation_config','artifact_sha256']
 missing=[k for k in required if k not in record]
 if missing: raise ValueError('missing identity fields: '+','.join(missing))
 return hashlib.sha256(json.dumps({k:record[k] for k in required},sort_keys=True,separators=(',',':')).encode()).hexdigest()
def index_records(records):
 out={}
 for r in records:
  key=canonical_identity(r)
  if key in out: raise ValueError('duplicate content-bound identity')
  out[key]=r
 return out
def reject_case_target_collision(records):
 seen={}
 for r in records:
  k=(r.get('case_id'),r.get('target'))
  if k in seen and canonical_identity(seen[k])!=canonical_identity(r): raise ValueError('ambiguous case_id/target collision')
  seen[k]=r
 return True
