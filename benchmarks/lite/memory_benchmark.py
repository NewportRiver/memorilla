"""Bounded LITE-004 persistence and quantization benchmark."""
from __future__ import annotations
import hashlib,json,shutil,tempfile,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
import sys; sys.path.insert(0,str(ROOT))
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.memory.quantization import decode_vector,encode_vector,cosine
from lite.runtime.preflight import ResourceLimits,preflight
from lite.runtime.telemetry import snapshot

def contract():
 return EncoderContract('sentence-transformers/all-MiniLM-L6-v2','1110a243fdf4706b3f48f1d95db1a4f5529b4d41','4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474','bert-wordpiece-vocab-fb140275','mean-mask-l2-v1')
def source(i,collection='actual'):
 text=f'Synthetic public evidence record {i} documents a bounded Android runtime result and preserves provenance.'
 return SourceRecord.from_content(f'{collection}-{i}',text,provenance='LITE-004 synthetic fixture',source_revision='1',access_scope='public',source_timestamp='2026-01-01',collection_id=collection,chunk_id='0',chunk_start=0,chunk_end=len(text))
def bytes_used(p):
 return sum(x.stat().st_size for x in p.parent.glob(p.name+'*') if x.exists())
def run():
 limits=ResourceLimits(min_available_ram_bytes=1024**3,abort_available_ram_bytes=768*1024**2,target_rss_bytes=192*1024**2,max_rss_bytes=256*1024**2,max_storage_bytes=128*1024**2,max_threads=2,max_concurrent_processes=1,timeout_seconds=30)
 pre=preflight(Path.home(),limits,0)
 if not pre['ok']: return {'status':'blocked','preflight':pre}
 base=Path.home()/'.cache/memorilla-lite/lite004-bench'; base.mkdir(parents=True,exist_ok=True); start=time.monotonic_ns(); out={'status':'passed','preflight':pre,'group_a':{},'group_b':{}}
 try:
  enc=MiniLMEncoder(max_length=128,threads=2); c=contract(); rows=[source(i) for i in range(100)]; lookup={x.evidence_id:x for x in rows}; calls=[0]
  def encode(text): calls[0]+=1; return enc.encode(text)
  for dtype in ('float32','float16','int8'):
   d=Path(tempfile.mkdtemp(prefix='group-a-',dir=base)); db=d/'memory.sqlite'; s=MemoryStore(db,lookup.get); t=time.perf_counter_ns(); first=[s.ingest(x,c,encode,dtype) for x in rows]; ingest_ms=(time.perf_counter_ns()-t)/1e6; t=time.perf_counter_ns(); dup=[s.ingest(x,c,encode,dtype) for x in rows]; duplicate_ms=(time.perf_counter_ns()-t)/1e6; vectors=[s.get_vector(x['memory_id'])['vector'] for x in first]; raw=np.stack([enc.encode(x.content) for x in rows]); restored=np.stack(vectors); sims=[cosine(raw[i],restored[i]) for i in range(100)]; q=raw[:5]@raw.T; qi=raw[:5]@restored.T; overlap=[len(set(np.argsort(-q[i])[:5])&set(np.argsort(-qi[i])[:5]))/5 for i in range(5)]; s.close(); out['group_a'][dtype]={'records':100,'ingest_ms':ingest_ms,'duplicate_ms':duplicate_ms,'encoder_calls_for_ingest_and_duplicate_and_measurement':calls[0],'database_bytes':bytes_used(db),'vector_payload_bytes':s.stats()['vector_bytes'] if False else sum(len(encode_vector(v,dtype,384)[0]) for v in raw),'mean_cosine':float(np.mean(sims)),'min_cosine':float(np.min(sims)),'top5_overlap_mean':float(np.mean(overlap)),'top5_overlap_min':float(np.min(overlap)),'duplicate_all_skipped':all(not x['encoded'] for x in dup)}; shutil.rmtree(d,ignore_errors=True)
  enc.close();
  for n in (100,1000,5000,10000,25000):
   d=Path(tempfile.mkdtemp(prefix=f'group-b-{n}-',dir=base)); db=d/'memory.sqlite'; rr={}; s=MemoryStore(db,rr.get); cc=EncoderContract('synthetic','v1','b'*64,'synthetic-tokenizer','synthetic-prep'); t=time.perf_counter_ns()
   for i in range(n):
    text=f'numeric synthetic record {i}'; rr[f'b-{i}']=SourceRecord.from_content(f'b-{i}',text,provenance='numeric',source_revision='1',access_scope='public',source_timestamp='t',collection_id='numeric'); s.ingest(rr[f'b-{i}'],cc,lambda _t,i=i: np.full(384,np.float32((i%127)-63)/63,dtype=np.float32),'int8')
   insert_ms=(time.perf_counter_ns()-t)/1e6; before=snapshot(f'before-{n}',start,Path.home()).as_dict(); t=time.perf_counter_ns(); count=sum(1 for _ in s.iter_vectors('numeric','public',512,cc)); iterate_ms=(time.perf_counter_ns()-t)/1e6; s.close(); size=bytes_used(db); out['group_b'][str(n)]={'records':n,'insert_ms':insert_ms,'iterate_ms':iterate_ms,'iterated':count,'database_bytes_with_wal':size,'raw_vector_bytes':n*384,'overhead_bytes':size-n*384,'peak_rss_bytes':before['peak_rss_bytes'],'available_ram_bytes':before['available_ram_bytes']}; shutil.rmtree(d,ignore_errors=True)
 finally:
  shutil.rmtree(base,ignore_errors=True)
 return out
if __name__=='__main__': print(json.dumps(run(),sort_keys=True))
