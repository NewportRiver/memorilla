from __future__ import annotations
from .evidence import EvidenceVerifier,EvidencePacket
class FallbackController:
 def __init__(self,lexical): self.lexical=lexical
 def retrieve(self,query,**kwargs):
  reason=kwargs.pop('fallback_reason','semantic_unavailable'); out=self.lexical.search(query,**kwargs); out['fallback_used']=True; out['fallback_reason']=reason; return out
class RetrievalController:
 def __init__(self,lexical,semantic,hybrid,catalog): self.lexical=lexical; self.semantic=semantic; self.hybrid=hybrid; self.verifier=EvidenceVerifier(catalog); self.fallback=FallbackController(lexical)
 def retrieve(self,query,method='semantic',collection_id=None,access_scope=None,as_of=None,top_k=5,allow_fallback=True):
  if not access_scope: return {'status':'abstain','reason':'access_scope_required','results':[],'omitted':[]}
  if top_k not in (1,3,5,10): raise ValueError('top_k must be 1, 3, 5, or 10')
  started=__import__('time').perf_counter_ns(); raw=None; fallback=False; reason=None
  try:
   if method=='lexical': raw=self.lexical.search(query,collection_id,access_scope,as_of,top_k)
   elif method=='semantic': raw=self.semantic.search(query,collection_id,access_scope,as_of,top_k)
   elif method=='hybrid': raw=self.hybrid.search(query,collection_id=collection_id,access_scope=access_scope,as_of=as_of,limit=top_k)
   else: raise ValueError('unknown retrieval method')
  except Exception as exc:
   if not allow_fallback or method=='lexical': return {'status':'abstain','reason':f'retrieval_failure:{type(exc).__name__}','results':[],'omitted':[]}
   fallback=True; reason=f'{method}_unavailable:{type(exc).__name__}'; raw=self.fallback.retrieve(query,collection_id=collection_id,access_scope=access_scope,as_of=as_of,limit=top_k,fallback_reason=reason)
  if method!='lexical' and allow_fallback and not raw.get('results'):
   fallback=True; reason=f'{method}_no_compatible_candidates'; raw=self.fallback.retrieve(query,collection_id=collection_id,access_scope=access_scope,as_of=as_of,limit=top_k,fallback_reason=reason)
  verified=[]; omitted=[]; seen=set()
  for candidate in raw['results']:
   if candidate['evidence_id'] in seen: continue
   seen.add(candidate['evidence_id']); item,why=self.verifier.verify(candidate,access_scope,collection_id,as_of)
   if item: item['retrieval_method']='lexical' if fallback else method; verified.append(item)
   else: omitted.append({'evidence_id':candidate['evidence_id'],'reason':why})
  status='ok' if verified else 'abstain'; return {'status':status,'reason':reason if fallback else (None if verified else 'insufficient_evidence'),'results':verified,'omitted':omitted,'fallback_used':fallback,'telemetry':{**raw.get('telemetry',{}),'evidence_resolution_operations':len(raw['results']),'complete_pipeline_ms':(__import__('time').perf_counter_ns()-started)/1e6},'evidence_packet':EvidencePacket(verified).as_dict()}
