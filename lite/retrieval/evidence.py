from __future__ import annotations
import re,sqlite3
class EvidenceCatalog:
 def __init__(self,records):
  self.records={r.evidence_id:r for r in records}; self.db=sqlite3.connect(':memory:'); self.db.execute('CREATE VIRTUAL TABLE evidence_fts USING fts5(evidence_id UNINDEXED,content,collection_id UNINDEXED,access_scope UNINDEXED,source_timestamp UNINDEXED)'); self.db.executemany('INSERT INTO evidence_fts VALUES (?,?,?,?,?)',[(r.evidence_id,r.content,r.collection_id,r.access_scope,r.source_timestamp) for r in records]); self.db.commit()
 def close(self): self.db.close()
 def resolve(self,evidence_id): return self.records.get(evidence_id)
 def eligible(self,collection_id=None,access_scope=None,as_of=None): return {r.evidence_id for r in self.records.values() if (collection_id is None or r.collection_id==collection_id) and (access_scope is None or r.access_scope==access_scope) and (as_of is None or r.source_timestamp<=as_of)}
 def lexical(self,query,eligible_ids=None,limit=100):
  terms=re.findall(r'[A-Za-z0-9_]+',query.lower());
  if not terms:return []
  rows=self.db.execute('SELECT evidence_id,bm25(evidence_fts) FROM evidence_fts WHERE evidence_fts MATCH ?',(' OR '.join(terms),)).fetchall(); allowed=eligible_ids; out=[(eid,float(-bm)) for eid,bm in rows if allowed is None or eid in allowed]; out.sort(key=lambda x:(-x[1],x[0])); return out[:limit]
 def excerpt(self,evidence_id,max_chars=320):
  r=self.resolve(evidence_id); return None if r is None else r.content[:max_chars]
class EvidenceVerifier:
 def __init__(self,catalog): self.catalog=catalog
 def verify(self,candidate,access_scope=None,collection_id=None,as_of=None,max_chars=320):
  r=self.catalog.resolve(candidate['evidence_id'])
  if r is None:return None,'missing_source'
  if access_scope is not None and r.access_scope!=access_scope:return None,'scope_violation'
  if collection_id is not None and r.collection_id!=collection_id:return None,'collection_mismatch'
  if as_of is not None and r.source_timestamp>as_of:return None,'temporal_leakage'
  if candidate.get('source_hash') and candidate['source_hash']!=r.source_hash:return None,'source_hash_mismatch'
  return {'evidence_id':r.evidence_id,'source_revision':r.source_revision,'source_hash':r.source_hash,'source_excerpt':r.content[:max_chars],'collection_id':r.collection_id,'access_scope':r.access_scope,'source_timestamp':r.source_timestamp,'retrieval_score':candidate.get('score'),'retrieval_rank':candidate.get('rank'),'verification_status':'verified'},None
class EvidencePacket:
 def __init__(self,results): self.results=results
 def as_dict(self):
  text='\n'.join(x['source_excerpt'] for x in self.results); return {'references':[x['evidence_id'] for x in self.results],'text':text,'character_count':len(text),'byte_count':len(text.encode()),'duplicate_excerpt_count':len(self.results)-len(set(x['source_excerpt'] for x in self.results))}
