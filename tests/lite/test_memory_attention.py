import sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.memory.iterator import vector_batches
from lite.attention.reference import attention,attention_stream,deterministic_weights

def test_persisted_vectors_feed_global_stream_attention(tmp_path):
 rows=[]
 for i in range(300):
  text=f'attention evidence {i}'; rows.append(SourceRecord.from_content(f'e{i}',text,provenance='fixture',source_revision='1',access_scope='public',source_timestamp='t',collection_id='c'))
 lookup={r.evidence_id:r for r in rows}; s=MemoryStore(tmp_path/'m.db',lookup.get); c=EncoderContract('mini','1','a'*64,'tok','prep')
 def enc(t): return np.random.default_rng(int(hashlib.sha256(t.encode()).hexdigest()[:8],16)).standard_normal(384,dtype=np.float32)
 for r in rows:s.ingest(r,c,enc,'float32')
 chunks=[x for x,_ in vector_batches(s,'c','public',128,c)]; memory=np.concatenate(chunks); q=enc('query'); w=deterministic_weights()
 class Restartable:
  def __iter__(self): return (x for x,_ in vector_batches(s,'c','public',128,c))
 expected=attention(memory,q,w,384,8,128); streamed=attention_stream(Restartable(),q,w,384,8)
 assert expected.shape==(8,384); assert np.isfinite(streamed).all(); assert np.max(np.abs(expected-streamed))<2e-5; s.close()
