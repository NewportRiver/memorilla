from __future__ import annotations
class HybridRetriever:
 def __init__(self,lexical,semantic,rrf_k=60): self.lexical=lexical; self.semantic=semantic; self.rrf_k=rrf_k
 def search(self,query,**kwargs):
  limit=kwargs.pop('limit',10); a=self.lexical.search(query,limit=limit,**kwargs); b=self.semantic.search(query,limit=limit,**kwargs); scores={}; sources={}
  for group in (a['results'],b['results']):
   for row in group: scores[row['evidence_id']]=scores.get(row['evidence_id'],0.0)+1.0/(self.rrf_k+row['rank']); sources.setdefault(row['evidence_id'],[]).append(row)
  order=sorted(scores,key=lambda x:(-scores[x],x)); results=[{'evidence_id':eid,'score':scores[eid],'rank':i+1,'component_ranks':sources[eid]} for i,eid in enumerate(order[:limit])]
  return {'results':results,'telemetry':{'lexical_candidates':len(a['results']),'semantic_candidates':len(b['results']),'rrf_k':self.rrf_k,'database_passes':2}}
