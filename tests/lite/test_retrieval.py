import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.retrieval import EvidenceCatalog,LexicalRetriever,SemanticRetriever,HybridRetriever,RetrievalController

def setup(tmp_path):
 rows=[SourceRecord.from_content('a','The verified restart checksum was preserved.',provenance='test',source_revision='1',access_scope='public',source_timestamp='2026-01-01',collection_id='c'),SourceRecord.from_content('b','The first dependency attempt failed.',provenance='test',source_revision='1',access_scope='public',source_timestamp='2026-01-02',collection_id='c'),SourceRecord.from_content('c','The pinned dependency corrected the failure.',provenance='test',source_revision='1',access_scope='public',source_timestamp='2026-01-03',collection_id='c')]
 cat=EvidenceCatalog(rows); s=MemoryStore(tmp_path/'m.sqlite',cat.resolve); contract=EncoderContract('test','r1','a'*64,'tok','prep')
 vec={'a':np.r_[np.ones(1),np.zeros(383)].astype(np.float32),'b':np.r_[np.zeros(1),np.ones(1),np.zeros(382)].astype(np.float32),'c':np.r_[np.zeros(2),np.ones(1),np.zeros(381)].astype(np.float32)}
 for r in rows:s.ingest(r,contract,lambda text, r=r:vec[r.evidence_id],'float16')
 enc=lambda text: vec['a'] if 'checksum' in text else vec['b']
 lex=LexicalRetriever(cat); sem=SemanticRetriever(s,enc,contract); return rows,cat,s,contract,RetrievalController(lex,sem,HybridRetriever(lex,sem),cat)

def test_three_methods_verify_evidence(tmp_path):
 rows,cat,s,c,ctl=setup(tmp_path)
 for method in ('lexical','semantic','hybrid'):
  out=ctl.retrieve('verified restart checksum',method=method,collection_id='c',access_scope='public',top_k=5); assert out['status']=='ok'; assert out['results'][0]['verification_status']=='verified'; assert out['results'][0]['source_hash']==rows[0].source_hash
 s.close();cat.close()

def test_temporal_scope_and_fallback(tmp_path):
 rows,cat,s,c,ctl=setup(tmp_path)
 out=ctl.retrieve('pinned dependency corrected',method='semantic',collection_id='c',access_scope='public',as_of='2026-01-02',top_k=3); assert all(x['evidence_id']!='c' for x in out['results']); assert all(x['source_timestamp']<='2026-01-02' for x in out['results'])
 s.close(); out=ctl.retrieve('verified restart checksum',method='semantic',collection_id='c',access_scope='public',top_k=3); assert out['fallback_used'] is True; assert out['results'][0]['verification_status']=='verified'; cat.close()

def test_required_scope_and_topk_validation(tmp_path):
 _,cat,s,_,ctl=setup(tmp_path)
 assert ctl.retrieve('x',method='semantic',collection_id='c',access_scope=None)['reason']=='access_scope_required'
 with pytest.raises(ValueError):ctl.retrieve('x',method='semantic',collection_id='c',access_scope='public',top_k=2)
 s.close();cat.close()
