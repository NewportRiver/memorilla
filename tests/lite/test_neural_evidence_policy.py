import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from lite.memory.provenance import SourceRecord
from lite.retrieval.evidence import EvidenceCatalog,EvidenceVerifier
from lite.neural.scoring import policy_rank

def rec(e,content,scope='public',ts='2026-01-01'):
 return SourceRecord.from_content(e,content,provenance='synthetic',source_revision='r1',access_scope=scope,source_timestamp=ts,collection_id='c',chunk_id=e,chunk_start=0,chunk_end=len(content))
def test_policy_filters_scope_and_time_before_ranking():
 rs=[rec('ok','authorized historical fact'),rec('private','private fact','private'),rec('future','future fact','public','2027-01-01')]; cat=EvidenceCatalog(rs); verifier=EvidenceVerifier(cat); slots=np.zeros((8,384),np.float32); docs=np.zeros((3,384),np.float32); slots[:,0]=1; docs[:,0]=1
 out=policy_rank(slots,docs,['private','future','ok'],verifier,'c','public','2026-06-01',top_k=2); assert [x['evidence_id'] for x in out['results']]==['ok']; assert all(x['verification_status']=='verified' for x in out['results']); cat.close()
def test_policy_empty_candidates_abstains():
 r=rec('ok','fact'); cat=EvidenceCatalog([r]); out=policy_rank(np.ones((8,384),np.float32),np.ones((0,384),np.float32),[],EvidenceVerifier(cat),'c','public',top_k=5); assert out['abstained'] and out['results']==[]; cat.close()
