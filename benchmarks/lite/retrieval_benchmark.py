"""LITE-005 bounded retrieval/evaluation benchmark; no holdout labels."""
from __future__ import annotations
import hashlib,json,statistics,tempfile,time,shutil
from pathlib import Path
import sys
ROOT_REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT_REPO))
import numpy as np
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.encoder.manifest import MODEL_REPO,MODEL_REVISION,MODEL_SHA256
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.memory.quantization import cosine
from lite.retrieval import EvidenceCatalog,LexicalRetriever,SemanticRetriever,HybridRetriever,RetrievalController
from lite.attention.reference import attention_stream
from lite.runtime.preflight import ResourceLimits,assert_preflight
from lite.runtime.telemetry import rss_bytes
ROOT=Path(__file__).resolve().parent/'fixtures'

def contract(): return EncoderContract(MODEL_REPO,MODEL_REVISION,MODEL_SHA256,'MiniLMTokenizer-basic-v1','attention-mask-mean-l2-v1',384)
def load_fixture():
 raw=json.loads((ROOT/'development_sources.json').read_text()); records=[SourceRecord.from_content(x['evidence_id'],x['content'],provenance=x['provenance'],source_revision=x['source_revision'],access_scope=x['access_scope'],source_timestamp=x['source_timestamp'],collection_id=x['collection_id'],chunk_id=x['chunk_id'],chunk_start=x['chunk_start'],chunk_end=x['chunk_end']) for x in raw['sources']]
 queries=json.loads((ROOT/'development_queries.json').read_text())['queries']; return records,queries

def metrics(rows,queries,method,k=5):
 bycat={}; totals={'queries':len(queries),'recall':0.0,'precision':0.0,'mrr':0.0,'multi_coverage':0.0,'complete_support':0,'expected_abstentions':0,'correct_abstentions':0,'scope_violations':0,'temporal_leakage':0,'invalid_returns':0}
 for q in queries:
  out=rows[q['query_id']]; ids=[x['evidence_id'] for x in out.get('results',[])][:k]; req=set(q['required_evidence_ids']); hit=len(req.intersection(ids)); expected=q['expected_abstention']; totals['recall']+=hit/len(req) if req else 0; totals['precision']+=hit/max(len(ids),1); totals['mrr']+=next((1/(i+1) for i,e in enumerate(ids) if e in req),0); totals['multi_coverage']+=hit/len(req) if req else (1.0 if not ids else 0.0); totals['complete_support']+=int(req.issubset(ids)) if req else int(not ids); totals['expected_abstentions']+=int(expected); totals['correct_abstentions']+=int(expected and not ids)
 for qid,out in rows.items():
  for x in out.get('omitted',[]): totals['scope_violations']+=int(x['reason']=='scope_violation'); totals['temporal_leakage']+=int(x['reason']=='temporal_leakage')
 n=len(queries); totals.update({'recall@'+str(k):totals.pop('recall')/n,'precision@'+str(k):totals.pop('precision')/n,'mrr':totals['mrr']/n,'multi_evidence_coverage':totals.pop('multi_coverage')/n,'complete_support_rate':totals.pop('complete_support')/n,'abstention_correct_rate':totals['correct_abstentions']/max(totals['expected_abstentions'],1)})
 for q in queries:
  cat=q['category']; bycat.setdefault(cat,[]).append(rows[q['query_id']])
 totals['by_category']={}
 for cat in sorted({q['category'] for q in queries}):
  qs=[q for q in queries if q['category']==cat]; rec=[]; complete=[]; rr=[]
  for q in qs:
   ids=[x['evidence_id'] for x in rows[q['query_id']].get('results',[])][:k]; req=set(q['required_evidence_ids']); hit=len(req&set(ids)); rec.append(hit/len(req) if req else (1.0 if not ids else 0.0)); complete.append(int(req.issubset(ids)) if req else int(not ids)); rr.append(next((1/(i+1) for i,e in enumerate(ids) if e in req),0))
  totals['by_category'][cat]={'queries':len(qs),'recall@'+str(k):sum(rec)/len(rec),'complete_support_rate':sum(complete)/len(complete),'mrr':sum(rr)/len(rr)}
 return totals

def build_store(path,records,enc,dtype):
 catalog=EvidenceCatalog(records); store=MemoryStore(path,catalog.resolve); c=contract(); cache={}
 def encode(text):
  if text not in cache: cache[text]=enc.encode(text)
  return cache[text]
 for r in records: store.ingest(r,c,encode,dtype)
 return store,catalog,c,cache

def run():
 pre=assert_preflight(path=Path.home(),limits=ResourceLimits(min_available_ram_bytes=1024*1024*1024,abort_available_ram_bytes=768*1024*1024,target_rss_bytes=192*1024*1024,max_rss_bytes=256*1024*1024,max_storage_bytes=128*1024*1024,max_threads=2,max_concurrent_processes=1,timeout_seconds=30),requested_storage_bytes=100*1024*1024)
 records,queries=load_fixture(); enc=MiniLMEncoder(threads=2); c=contract(); results={'protocol':{'seed':20261120,'queries':len(queries),'sources':len(records),'top_k':[1,3,5,10],'rrf_k':60,'default_dtype':'float16'},'preflight':pre,'methods':{},'quantization':{},'attention':{},'context':{}}
 with tempfile.TemporaryDirectory(prefix='lite005-',dir=str(Path.home()/'.cache')) as td:
  stores={}; cats={}; caches={}; semantic_outputs={}; context_samples=[]
  for dtype in ('float32','float16','int8'):
   s,cat,_,cache=build_store(Path(td)/f'{dtype}.sqlite',records,enc,dtype); stores[dtype]=s; cats[dtype]=cat; caches[dtype]=cache
  for dtype in ('float32','float16','int8'):
   s,cat,_,_=stores[dtype],cats[dtype],c,None
   sem=SemanticRetriever(s,enc,c,dtype=dtype); con=RetrievalController(LexicalRetriever(cat),sem,HybridRetriever(LexicalRetriever(cat),sem),cat)
   for method in ('lexical','semantic','hybrid'):
    if dtype!='float16' and method!='semantic': continue
    out={}; samples=[]
    for q in queries:
     x=con.retrieve(q['query'],method=method,collection_id=q['scenario_id'],access_scope=q['access_scope'],as_of=q['historical_cutoff'],top_k=5); out[q['query_id']]=x; samples.append(x['telemetry'].get('complete_pipeline_ms',0))
     if dtype=='float16' and method=='semantic':
      raw_chars=sum(len(r.content) for r in records if r.collection_id==q['scenario_id'] and r.access_scope==q['access_scope'] and (q['historical_cutoff'] is None or r.source_timestamp<=q['historical_cutoff'])); packet=x.get('evidence_packet',{}); context_samples.append({'raw_chars':raw_chars,'retrieved_chars':packet.get('character_count',0),'references':len(packet.get('references',[]))})
    if method=='semantic': semantic_outputs[dtype]=out
    results['methods'][f'{method}_{dtype}']={'metrics':metrics(out,queries,method,5),'latency_ms':{'samples':samples,'median':statistics.median(samples),'count':len(samples)},'max_rss':rss_bytes()}
  # Quantization ranking and required-evidence comparison against float32.
  ref=semantic_outputs['float32']
  for dtype in ('float16','int8'):
   overlaps=[]; eligible_overlaps=[]; deltas=[]; worst=[]
   for q in queries:
    a=[x['evidence_id'] for x in ref[q['query_id']]['results'][:5]]; b=[x['evidence_id'] for x in semantic_outputs[dtype][q['query_id']]['results'][:5]]; ov=len(set(a)&set(b))/max(len(a),1); overlaps.append(ov); eligible_overlaps.append(ov) if (a or b) else None; req=set(q['required_evidence_ids']); ra=len(req&set(a))/len(req) if req else 1.0; rb=len(req&set(b))/len(req) if req else 1.0; deltas.append(rb-ra); worst.append({'query_id':q['query_id'],'overlap':ov,'required_recall_delta':deltas[-1]})
   results['quantization'][dtype]={'mean_top5_overlap_eligible':statistics.mean(eligible_overlaps) if eligible_overlaps else 1.0,'minimum_top5_overlap':min(overlaps),'required_evidence_recall_delta_mean':statistics.mean(deltas),'required_evidence_recall_delta_min':min(deltas),'worst_queries':sorted(worst,key=lambda x:(x['overlap'],x['query_id']))[:5]}
   rows=list(stores[dtype].iter_vectors(access_scope='public',contract=c)); sims=[float(cosine(caches[dtype][next(r.content for r in records if r.evidence_id==item['evidence_id'])],item['vector'])) for item in rows]
   results['quantization'][dtype]['mean_reconstructed_cosine']=statistics.mean(sims); results['quantization'][dtype]['minimum_reconstructed_cosine']=min(sims); results['quantization'][dtype]['eligible_query_count']=sum(bool(semantic_outputs['float32'][q['query_id']]['results']) or bool(semantic_outputs[dtype][q['query_id']]['results']) for q in queries)

  results['context']={'note':'characters/bytes, not model tokens','samples':context_samples,'median_raw_chars':statistics.median(x['raw_chars'] for x in context_samples),'median_retrieved_chars':statistics.median(x['retrieved_chars'] for x in context_samples),'median_context_reduction':statistics.median(1-x['retrieved_chars']/max(x['raw_chars'],1) for x in context_samples),'median_references':statistics.median(x['references'] for x in context_samples)}
  attention_rows=[]
  rng=np.random.default_rng(20261120); q=enc.encode('query for attention resource comparison')
  for n in (100,1000,5000):
   vecs=rng.standard_normal((n,384),dtype=np.float32); vecs/=np.maximum(np.linalg.norm(vecs,axis=1,keepdims=True),1e-12); chunks=[vecs[i:i+128] for i in range(0,n,128)]
   class Restartable:
    def __iter__(self): return iter(chunks)
   t0=time.perf_counter_ns(); out=attention_stream(Restartable(),q,dim=384,slots=8); elapsed=(time.perf_counter_ns()-t0)/1e6
   attention_rows.append({'records':n,'attention_ms':elapsed,'rss_bytes':rss_bytes(),'passes':2,'output_shape':list(out.shape),'finite':bool(np.isfinite(out).all())})
  results['attention']=attention_rows; results['rss_peak']=rss_bytes()
 for s in stores.values(): s.close()
 for cat in cats.values(): cat.close()
 enc.close(); return results
if __name__=='__main__': print(json.dumps(run(),sort_keys=True,indent=2))
