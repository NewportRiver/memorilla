import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lite.memory.provenance import SourceRecord
from lite.retrieval.evidence import EvidenceCatalog,EvidenceVerifier,EvidencePacket

def test_hash_scope_time_verification():
 r=SourceRecord.from_content('e','authoritative excerpt',provenance='test',source_revision='2',access_scope='public',source_timestamp='2026-01-02',collection_id='c'); cat=EvidenceCatalog([r]); v=EvidenceVerifier(cat)
 ok,why=v.verify({'evidence_id':'e','source_hash':r.source_hash,'score':.9,'rank':1},'public','c','2026-01-03'); assert why is None; assert ok is not None and ok['verification_status']=='verified'
 assert v.verify({'evidence_id':'e','source_hash':'bad'},'public','c')[1]=='source_hash_mismatch'
 assert v.verify({'evidence_id':'e'},'private','c')[1]=='scope_violation'
 assert v.verify({'evidence_id':'e'},'public','c','2026-01-01')[1]=='temporal_leakage'; assert v.verify({'evidence_id':'missing'},'public','c')[1]=='missing_source'; cat.close()

def test_packet_counts_and_dedup():
 r={'evidence_id':'e','source_excerpt':'same'}; p=EvidencePacket([r,r]).as_dict(); assert p['character_count']==9; assert p['duplicate_excerpt_count']==1; assert p['references']==['e','e']
