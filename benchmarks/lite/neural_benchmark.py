"""Bounded LITE-006 architecture benchmark; no training or holdout labels."""
from __future__ import annotations
import json,statistics,tempfile,time,resource,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.retrieval.evidence import EvidenceCatalog,EvidenceVerifier
from lite.neural import NeuralConfig,NumpyAssociativeMemory
from lite.neural.scoring import score_slots,rank_evidence
from lite.neural.attention import attention_agreement

def contract(): return EncoderContract('synthetic-neural-benchmark','r1','a'*64,'synthetic-tokenizer','synthetic-preprocess',384)
def timed(fn,reps=1):
 vals=[]; out=None
 for _ in range(reps):
  t=time.perf_counter_ns(); out=fn(); vals.append((time.perf_counter_ns()-t)/1e6)
 return out,{'cold_ms':vals[0],'warm_median_ms':statistics.median(vals[1:]) if len(vals)>1 else vals[0],'samples':vals}
def run():
 c=NeuralConfig(); t_init=time.perf_counter_ns(); model=NumpyAssociativeMemory(c); init_ms=(time.perf_counter_ns()-t_init)/1e6; rng=np.random.default_rng(c.seed); q=rng.normal(size=384).astype('float32'); results={'status':'passed','config':c.as_dict(),'parameter_count':model.parameter_count(),'weight_bytes':sum(x.nbytes for x in model.state.values()),'initialization_ms':init_ms,'sizes':[],'resource':{'max_rss_bytes':0,'max_threads':2,'training':'not_run'}}
 with tempfile.TemporaryDirectory(prefix='lite006-',dir=str(Path.home()/'.cache')) as td:
  for n in (100,1000,5000):
   vec=rng.normal(size=(n,384)).astype('float32'); vec/=np.maximum(np.linalg.norm(vec,axis=1,keepdims=True),1e-8); records=[SourceRecord.from_content(f'n{n}-{i}',f'synthetic benchmark evidence {n} {i}',provenance='synthetic',source_revision='r1',access_scope='public',source_timestamp='2026-01-01',collection_id='bench',chunk_id=str(i),chunk_start=0,chunk_end=32) for i in range(n)]; cat=EvidenceCatalog(records); mapping={r.content:vec[i] for i,r in enumerate(records)}; store=MemoryStore(Path(td)/f'{n}.sqlite',cat.resolve); cc=contract();
   for r in records: store.ingest(r,cc,lambda text,m=mapping:m[text],'float16')
   def traverse(): return list(store.iter_vectors('bench','public',128,cc))
   t=time.perf_counter_ns(); rows=traverse(); db_ms=(time.perf_counter_ns()-t)/1e6; db_warm=[]
   for _ in range(3): t=time.perf_counter_ns(); traverse(); db_warm.append((time.perf_counter_ns()-t)/1e6)
   db_warm_ms=statistics.median(db_warm); arr=np.stack([x['vector'] for x in rows]).astype('float32'); ids=[x['evidence_id'] for x in rows];
   t=time.perf_counter_ns(); out=model.forward(q,arr); matrix_ms=(time.perf_counter_ns()-t)/1e6; _,warm=timed(lambda:model.forward(q,arr),4); stream_chunks=[(arr[i:i+128],None) for i in range(0,n,128)]; t=time.perf_counter_ns(); stream=model.forward_stream(q,stream_chunks); stream_ms=(time.perf_counter_ns()-t)/1e6; agreement=attention_agreement(model,q,arr); t=time.perf_counter_ns(); scores=score_slots(out,arr); ranked=rank_evidence(scores,ids,top_k=5); score_ms=(time.perf_counter_ns()-t)/1e6; verifier=EvidenceVerifier(cat); t=time.perf_counter_ns(); verified=[verifier.verify(x,'public','bench','2026-01-01')[0] for x in ranked]; verify_ms=(time.perf_counter_ns()-t)/1e6; verify_warm=[]
   for _ in range(3): t=time.perf_counter_ns(); [verifier.verify(x,'public','bench','2026-01-01')[0] for x in ranked]; verify_warm.append((time.perf_counter_ns()-t)/1e6)
   verify_warm_ms=statistics.median(verify_warm); results['sizes'].append({'records':n,'initialization_ms':init_ms,'query_conditioned_full_ms':matrix_ms,'query_conditioned_warm_median_ms':warm['warm_median_ms'],'streamed_cross_attention_ms':stream_ms,'query_projection_note':'included in query-conditioned pipeline; no duplicate projection timing claimed','memory_projection_note':'included in attention pipeline; SQLite traversal measured separately','evidence_scoring_ms':score_ms,'sqlite_traversal_ms':db_ms,'sqlite_traversal_warm_median_ms':db_warm_ms,'evidence_verification_ms':verify_ms,'evidence_verification_warm_median_ms':verify_warm_ms,'collection_passes_streamed':2,'verified_results':sum(x is not None for x in verified),'stream_full_max_abs_error':agreement['max_abs_error'],'stream_full_allclose':agreement['allclose'],'output_shape':list(out.shape),'finite':bool(np.isfinite(out).all()),'database_stats':store.stats(),'rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}); results['resource']['max_rss_bytes']=max(results['resource']['max_rss_bytes'],results['sizes'][-1]['rss_bytes']); store.close(); cat.close()
  return results
if __name__=='__main__': print(json.dumps(run(),sort_keys=True,indent=2))
