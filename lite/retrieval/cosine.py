from __future__ import annotations
import heapq,time
import numpy as np
from lite.memory.quantization import cosine
class SemanticRetriever:
 def __init__(self,store,encoder,contract,dtype='float16',chunk_size=128): self.store=store; self.encoder=encoder; self.contract=contract; self.dtype=dtype; self.chunk_size=chunk_size
 def search(self,query,collection_id=None,access_scope=None,as_of=None,limit=10):
  if limit not in (1,3,5,10): raise ValueError('limit must be 1, 3, 5, or 10')
  started=time.perf_counter_ns(); qstart=time.perf_counter_ns(); q=self.encoder(query); query_ms=(time.perf_counter_ns()-qstart)/1e6; validated=0; heap=[]; order=0
  for item in self.store.iter_vectors(collection_id,access_scope,self.chunk_size,self.contract,as_of):
   score=cosine(q,item['vector']); key=(score,-order,item); order+=1; validated+=1
   if len(heap)<limit: heapq.heappush(heap,key)
   elif key[:2]>heap[0][:2]: heapq.heapreplace(heap,key)
  rows=sorted(heap,key=lambda x:(-x[0],x[2]['evidence_id']))
  return {'results':[{'evidence_id':x[2]['evidence_id'],'memory_id':x[2]['memory_id'],'source_hash':x[2]['source_hash'],'source_revision':x[2]['source_revision'],'source_timestamp':x[2]['source_timestamp'],'access_scope':x[2]['access_scope'],'collection_id':x[2]['collection_id'],'score':float(x[0]),'rank':i+1,'dtype':x[2]['dtype']} for i,x in enumerate(rows)],'telemetry':{'query_encoding_ms':query_ms,'candidate_scan_ms':(time.perf_counter_ns()-started)/1e6,'validated_vectors':validated,'database_passes':1}}
