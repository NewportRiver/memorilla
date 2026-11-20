from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib,json,time
@dataclass(frozen=True)
class SourceRecord:
 evidence_id:str; content:str; source_hash:str; provenance:str; source_revision:str; access_scope:str; source_timestamp:str; collection_id:str; chunk_id:str='0'; chunk_start:int=0; chunk_end:int|None=None
 @classmethod
 def from_content(cls,evidence_id,content,**kwargs): return cls(evidence_id,content,hashlib.sha256(content.encode()).hexdigest(),**kwargs)
 def verify(self):
  actual=hashlib.sha256(self.content.encode()).hexdigest()
  if actual!=self.source_hash: raise ValueError('source hash mismatch')
  if not self.evidence_id or not self.access_scope: raise ValueError('evidence_id and access_scope are required')
  return True
@dataclass(frozen=True)
class EncoderContract:
 model_identity:str; encoder_revision:str; artifact_sha256:str; tokenizer_identity:str; preprocessing_identity:str; dimension:int=384
 def canonical(self): return asdict(self)
 def key(self,source:SourceRecord):
  payload={'source_hash':source.source_hash,'evidence_id':source.evidence_id,'scope':source.access_scope,'collection':source.collection_id,'chunk':source.chunk_id,'chunk_start':source.chunk_start,'chunk_end':source.chunk_end,'contract':self.canonical()}
  return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def now_ns(): return time.time_ns()
