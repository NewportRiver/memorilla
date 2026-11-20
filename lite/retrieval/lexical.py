from __future__ import annotations
class LexicalRetriever:
 def __init__(self,catalog): self.catalog=catalog
 def search(self,query,collection_id=None,access_scope=None,as_of=None,limit=10):
  eligible=self.catalog.eligible(collection_id,access_scope,as_of); rows=self.catalog.lexical(query,eligible,limit)
  return {'results':[{'evidence_id':eid,'score':score,'rank':i+1} for i,(eid,score) in enumerate(rows)],'telemetry':{'validated_vectors':0,'database_passes':1}}
