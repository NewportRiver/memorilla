import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,pytest
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore

def contract(tag='r1'):
 return EncoderContract('MiniLM','1', 'a'*64, 'bert-wordpiece-v1','mean-mask-l2-v1')
def fixture(n=12,scope='public'):
 rows=[]
 for i in range(n):
  text=f'public synthetic evidence record {i} describes a bounded runtime result.'
  rows.append(SourceRecord.from_content(f'e{i}',text,provenance='synthetic-fixture',source_revision='r1',access_scope=scope,source_timestamp=f'2026-01-{i+1:02d}',collection_id='c1',chunk_id='0',chunk_start=0,chunk_end=len(text)))
 return rows
def resolver(rows):
 return {x.evidence_id:x for x in rows}
def enc(text):
 seed=int(hashlib.sha256(text.encode()).hexdigest()[:8],16); r=np.random.default_rng(seed); return r.standard_normal(384,dtype=np.float32)

def test_schema_ingest_idempotency_restart_and_scope(tmp_path):
 rows=fixture(); lookup=resolver(rows); db=tmp_path/'memory.sqlite'; s=MemoryStore(db,lookup.get); c=contract(); calls=[]
 def counted(x): calls.append(x); return enc(x)
 first=s.ingest(rows[0],c,counted,'float32'); second=s.ingest(rows[0],c,counted,'float32'); assert first['encoded'] and not second['encoded'] and len(calls)==1
 got=s.get_vector(first['memory_id'],scope='public',contract=c); assert got['vector'].shape==(384,)
 with pytest.raises(PermissionError): s.get_vector(first['memory_id'],scope='private')
 s.close(); s=MemoryStore(db,lookup.get); assert np.array_equal(s.get_vector(first['memory_id'])['vector'],got['vector']); assert s.stats()['valid']==1; s.close()

def test_source_change_supersedes_and_invalidation(tmp_path):
 old=fixture(1)[0]; lookup={old.evidence_id:old}; s=MemoryStore(tmp_path/'m.db',lookup.get); c=contract(); s.ingest(old,c,enc)
 new=SourceRecord.from_content(old.evidence_id,'changed evidence content',provenance=old.provenance,source_revision='r2',access_scope=old.access_scope,source_timestamp=old.source_timestamp,collection_id=old.collection_id,chunk_id='0',chunk_start=0,chunk_end=23); lookup[old.evidence_id]=new
 result=s.ingest(new,c,enc); assert result['encoded']; assert s.stats()['invalid']==1
 assert s.invalidate_source(old.evidence_id,source_hash=new.source_hash)==1; assert s.stats()['valid']==0
 s.close()

def test_deleted_source_and_model_invalidation_fail_closed(tmp_path):
 row=fixture(1)[0]; lookup={row.evidence_id:row}; s=MemoryStore(tmp_path/'m.db',lookup.get); c=contract(); mid=s.ingest(row,c,enc)['memory_id']; del lookup[row.evidence_id]
 with pytest.raises(ValueError,match='evidence missing'): s.get_vector(mid)
 row=fixture(1)[0]; lookup[row.evidence_id]=row; s2=MemoryStore(tmp_path/'m2.db',lookup.get); s2.ingest(row,c,enc); other=EncoderContract('mini','different','b'*64,'tok','prep'); assert s2.invalidate_incompatible(other)==1; assert s2.stats()['valid']==0; s2.close(); s.close()

def test_missing_hash_corruption_and_invalid_contract(tmp_path):
 row=fixture(1)[0]; s=MemoryStore(tmp_path/'m.db',resolver([row]).get); c=contract(); out=s.ingest(row,c,enc); mid=out['memory_id'];
 with s.db: s.db.execute('UPDATE embeddings SET vector_bytes=? WHERE memory_id=?',(b'bad',mid))
 with pytest.raises(ValueError): s.get_vector(mid)
 assert s.verify_integrity()['invalid']==1
 s.close(); missing=MemoryStore(tmp_path/'missing.db',lambda _id:None)
 with pytest.raises(ValueError,match='evidence missing'): missing.ingest(row,c,enc)
 missing.close()

def test_transaction_rollback_and_bounded_iteration(tmp_path):
 rows=fixture(300); s=MemoryStore(tmp_path/'m.db',resolver(rows).get); c=contract();
 def fail(_): raise RuntimeError('encoder failure')
 with pytest.raises(RuntimeError): s.ingest(rows[0],c,fail)
 assert s.stats()['total']==0
 for row in rows: s.ingest(row,c,enc,'int8')
 from lite.memory.iterator import iter_batches,vector_batches
 assert [len(x) for x in iter_batches(s,'c1','public',128,c)]==[128,128,44]
 assert sum(x.shape[0] for x,_ in vector_batches(s,'c1','public',256,c))==300
 assert s.stats()['vector_bytes']==300*384
 s.close()
