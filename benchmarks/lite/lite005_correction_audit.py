"""Audit LITE-005 without changing its historical benchmark artifacts."""
from __future__ import annotations
import json,sys,statistics
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from lite.encoder.onnx_encoder import MiniLMEncoder
from lite.encoder.manifest import MODEL_REPO,MODEL_REVISION,MODEL_SHA256
from lite.memory.provenance import SourceRecord,EncoderContract
from lite.memory.store import MemoryStore
from lite.retrieval import EvidenceCatalog,LexicalRetriever,SemanticRetriever,HybridRetriever,RetrievalController
FIX=ROOT/'benchmarks/lite/fixtures'
def contract(): return EncoderContract(MODEL_REPO,MODEL_REVISION,MODEL_SHA256,'MiniLMTokenizer-basic-v1','attention-mask-mean-l2-v1',384)
def run():
 raw=json.loads((FIX/'development_sources.json').read_text()); records=[SourceRecord.from_content(x['evidence_id'],x['content'],provenance=x['provenance'],source_revision=x['source_revision'],access_scope=x['access_scope'],source_timestamp=x['source_timestamp'],collection_id=x['collection_id'],chunk_id=x['chunk_id'],chunk_start=x['chunk_start'],chunk_end=x['chunk_end']) for x in raw['sources']]; queries=json.loads((FIX/'development_queries.json').read_text())['queries']; enc=MiniLMEncoder(threads=2); cat=EvidenceCatalog(records); store=MemoryStore(':memory:',cat.resolve); c=contract(); cache={}
 def encode(t): cache.setdefault(t,enc.encode(t)); return cache[t]
 for r in records: store.ingest(r,c,encode,'float16')
 sem=SemanticRetriever(store,enc,c,dtype='float16'); lex=LexicalRetriever(cat); con=RetrievalController(lex,sem,HybridRetriever(lex,sem),cat); outs={m:{} for m in ('lexical','semantic','hybrid')}; rows={m:[] for m in outs}
 for q in queries:
  for m in outs:
   o=con.retrieve(q['query'],method=m,collection_id=q['scenario_id'],access_scope=q['access_scope'],as_of=q['historical_cutoff'],top_k=5); outs[m][q['query_id']]=o; rows[m].append(o)
 report={'historical_benchmark_preserved':True,'queries':len(queries),'methods':{},'ranking_comparison':{},'context_recomputation':{},'holdout_independent_file_present':False}
 for m in outs:
  req_total=[]; hit=[]; complete=[]; expected=[]; correct=[]; enc_times=[]; verify_times=[]
  for q,o in zip(queries,rows[m]):
   ids=[x['evidence_id'] for x in o.get('results',[])]; req=set(q['required_evidence_ids']); eligible={r.evidence_id for r in records if r.collection_id==q['scenario_id'] and r.access_scope==q['access_scope'] and (q['historical_cutoff'] is None or r.source_timestamp<=q['historical_cutoff'])}; policy_expected=bool(q['expected_abstention']) or not (req & eligible)
   if req and req&eligible: req_total.append(q); hit.append(len(req&set(ids))/len(req)); complete.append(req.issubset(ids))
   expected.append(policy_expected); correct.append(policy_expected and not ids); enc_times.append(o.get('telemetry',{}).get('query_encoding_ms',0)); verify_times.append(o.get('telemetry',{}).get('complete_pipeline_ms',0))
  report['methods'][m]={'required_evidence_queries':len(req_total),'corrected_recall_required_only':statistics.mean(hit) if hit else None,'corrected_complete_support_required_only':statistics.mean(complete) if complete else None,'policy_abstention_expected':sum(expected),'policy_abstention_correct':sum(correct),'policy_abstention_accuracy':sum(correct)/max(sum(expected),1),'mean_query_encoding_ms':statistics.mean(enc_times),'mean_complete_pipeline_ms':statistics.mean(verify_times),'all_query_count':len(rows[m])}
 for a,b in (('lexical','semantic'),('lexical','hybrid'),('semantic','hybrid')):
  same=sum([ [x['evidence_id'] for x in outs[a][q['query_id']].get('results',[])]==[x['evidence_id'] for x in outs[b][q['query_id']].get('results',[])] for q in queries]); report['ranking_comparison'][a+'=='+b]={'identical_queries':same,'total_queries':len(queries),'identical_fraction':same/len(queries)}
 samples=[]
 for q,o in zip(queries,rows['semantic']):
  eligible=[r for r in records if r.collection_id==q['scenario_id'] and r.access_scope==q['access_scope'] and (q['historical_cutoff'] is None or r.source_timestamp<=q['historical_cutoff'])]; packet=o.get('evidence_packet',{}); raw=sum(len(r.content) for r in eligible); samples.append({'query_id':q['query_id'],'raw_chars':raw,'packet_chars':packet.get('character_count',0),'packet_references':len(packet.get('references',[])),'packet_exceeds_raw':packet.get('character_count',0)>raw})
 report['context_recomputation']={'samples':samples,'exceeds_count':sum(x['packet_exceeds_raw'] for x in samples),'note':'Raw is authoritative eligible source text; packet is newline-joined verified excerpts.'}
 store.close();cat.close();enc.close(); return report
if __name__=='__main__': print(json.dumps(run(),sort_keys=True,indent=2))
