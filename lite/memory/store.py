from __future__ import annotations
import sqlite3,time,hashlib
from pathlib import Path
from .schema import SCHEMA_SQL,SCHEMA_VERSION
from .provenance import SourceRecord,EncoderContract
from .quantization import encode_vector,decode_vector
class MemoryStore:
 def __init__(self,path,evidence_resolver=None):
  self.path=Path(path); self.path.parent.mkdir(parents=True,exist_ok=True); self.evidence_resolver=evidence_resolver
  self.db=sqlite3.connect(str(self.path),isolation_level=None); self.db.row_factory=sqlite3.Row; self.db.execute('PRAGMA foreign_keys=ON'); self.db.execute('PRAGMA journal_mode=WAL'); self.db.execute('PRAGMA synchronous=FULL'); self.db.executescript(SCHEMA_SQL)
  row=self.db.execute('SELECT version FROM schema_version ORDER BY version DESC LIMIT 1').fetchone()
  if row is None:self.db.execute('INSERT INTO schema_version(version,applied_at_ns) VALUES (?,?)',(SCHEMA_VERSION,time.time_ns()))
  elif row[0]!=SCHEMA_VERSION: raise RuntimeError(f'unsupported schema version {row[0]}')
 def close(self): self.db.close()
 def _resolve(self,source,scope=None):
  if self.evidence_resolver is None: raise ValueError('evidence resolver is required')
  found=self.evidence_resolver(source.evidence_id)
  if found is None: raise ValueError('evidence missing')
  r=found if isinstance(found,SourceRecord) else SourceRecord(**found); r.verify()
  if scope is not None and r.access_scope!=scope: raise PermissionError('cross-scope access attempt')
  if r.source_hash!=source.source_hash: raise ValueError('source hash mismatch')
  return r
 def _row_vector(self,row): return decode_vector(row['vector_bytes'],row['dtype'],row['dimension'],row['quantization_scale'],row['vector_checksum'])
 def ingest(self,source:SourceRecord,contract:EncoderContract,encode,dtype='float32'):
  source.verify(); self._resolve(source,source.access_scope); key=contract.key(source)
  existing=self.db.execute('SELECT memory_id,validity_status FROM embeddings WHERE idempotency_key=?',(key,)).fetchone()
  if existing and existing['validity_status']=='valid': return {'memory_id':existing['memory_id'],'encoded':False,'status':'existing'}
  vector=encode(source.content); raw,scale,checksum=encode_vector(vector,dtype,contract.dimension); scale=None if scale is None else float(scale); memory_id=hashlib.sha256((key+'|'+dtype).encode()).hexdigest()
  try:
   self.db.execute('BEGIN IMMEDIATE')
   self.db.execute("UPDATE embeddings SET validity_status='superseded',invalid_reason=? WHERE evidence_id=? AND access_scope=? AND chunk_id=? AND source_hash<>? AND validity_status='valid'",('source revision superseded',source.evidence_id,source.access_scope,source.chunk_id,source.source_hash))
   self.db.execute('''INSERT OR REPLACE INTO embeddings(memory_id,idempotency_key,evidence_id,source_hash,provenance,source_revision,access_scope,source_timestamp,ingestion_timestamp_ns,collection_id,chunk_id,chunk_start,chunk_end,model_identity,encoder_revision,artifact_sha256,tokenizer_identity,preprocessing_identity,dimension,dtype,quantization_method,quantization_scale,vector_bytes,vector_checksum,validity_status,invalid_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(memory_id,key,source.evidence_id,source.source_hash,source.provenance,source.source_revision,source.access_scope,source.source_timestamp, time.time_ns(),source.collection_id,source.chunk_id,source.chunk_start,source.chunk_end,contract.model_identity,contract.encoder_revision,contract.artifact_sha256,contract.tokenizer_identity,contract.preprocessing_identity,contract.dimension,dtype,'symmetric_per_vector' if dtype=='int8' else 'none',scale,sqlite3.Binary(raw),checksum,'valid',None))
   self.db.execute('COMMIT')
  except Exception:
   self.db.execute('ROLLBACK'); raise
  return {'memory_id':memory_id,'encoded':True,'status':'inserted','dtype':dtype}
 def get_vector(self,memory_id,scope=None,contract=None):
  row=self.db.execute('SELECT * FROM embeddings WHERE memory_id=?',(memory_id,)).fetchone()
  if row is None: raise KeyError('memory not found')
  if row['validity_status']!='valid': raise ValueError('memory is not valid')
  if scope is not None and row['access_scope']!=scope: raise PermissionError('cross-scope access attempt')
  if contract and (row['encoder_revision']!=contract.encoder_revision or row['artifact_sha256']!=contract.artifact_sha256 or row['tokenizer_identity']!=contract.tokenizer_identity or row['preprocessing_identity']!=contract.preprocessing_identity): raise ValueError('incompatible model or preprocessing configuration')
  source=SourceRecord(row['evidence_id'],'',row['source_hash'],row['provenance'],row['source_revision'],row['access_scope'],row['source_timestamp'],row['collection_id'],row['chunk_id'],row['chunk_start'],row['chunk_end']); self._resolve(source,row['access_scope'])
  return {'memory_id':row['memory_id'],'evidence_id':row['evidence_id'],'access_scope':row['access_scope'],'collection_id':row['collection_id'],'source_hash':row['source_hash'],'source_revision':row['source_revision'],'source_timestamp':row['source_timestamp'],'provenance':row['provenance'],'dtype':row['dtype'],'dimension':row['dimension'],'vector':self._row_vector(row),'validity_status':row['validity_status']}
 def iter_vectors(self,collection_id=None,access_scope=None,chunk_size=128,contract=None,as_of=None):
  if chunk_size not in (128,256,512): raise ValueError('chunk_size must be 128, 256, or 512')
  clauses=["validity_status='valid'"]; args=[]
  if collection_id is not None: clauses.append('collection_id=?'); args.append(collection_id)
  if access_scope is not None: clauses.append('access_scope=?'); args.append(access_scope)
  if as_of is not None: clauses.append('source_timestamp<=?'); args.append(as_of)
  if contract: clauses += ['encoder_revision=?','artifact_sha256=?','tokenizer_identity=?','preprocessing_identity=?']; args += [contract.encoder_revision,contract.artifact_sha256,contract.tokenizer_identity,contract.preprocessing_identity]
  cur=self.db.execute('SELECT * FROM embeddings WHERE '+' AND '.join(clauses)+' ORDER BY memory_id',args)
  for row in cur: yield self.get_vector(row['memory_id'],access_scope,contract)
 def invalidate_source(self,evidence_id,source_hash=None,reason='source invalidated'):
  if source_hash:return self.db.execute("UPDATE embeddings SET validity_status='invalid',invalid_reason=? WHERE evidence_id=? AND source_hash=? AND validity_status='valid'",(reason,evidence_id,source_hash)).rowcount
  return self.db.execute("UPDATE embeddings SET validity_status='invalid',invalid_reason=? WHERE evidence_id=? AND validity_status='valid'",(reason,evidence_id)).rowcount
 def invalidate_incompatible(self,contract,reason='encoder configuration invalidated'):
  return self.db.execute("UPDATE embeddings SET validity_status='invalid',invalid_reason=? WHERE (encoder_revision<>? OR artifact_sha256<>? OR tokenizer_identity<>? OR preprocessing_identity<>?) AND validity_status='valid'",(reason,contract.encoder_revision,contract.artifact_sha256,contract.tokenizer_identity,contract.preprocessing_identity)).rowcount
 def verify_integrity(self):
  rows=self.db.execute("SELECT * FROM embeddings WHERE validity_status IN ('valid','corrupt') ORDER BY memory_id").fetchall(); bad=[]
  for row in rows:
   try:self._row_vector(row)
   except Exception as e: bad.append({'memory_id':row['memory_id'],'error':str(e)})
  return {'checked':len(rows),'valid':len(rows)-len(bad),'invalid':len(bad),'errors':bad}
 def stats(self):
  row=self.db.execute("SELECT COUNT(*) AS total,SUM(validity_status='valid') AS valid,SUM(validity_status<>'valid') AS invalid,COALESCE(SUM(length(vector_bytes)),0) AS vector_bytes FROM embeddings").fetchone(); page=self.db.execute('PRAGMA page_count').fetchone()[0]*self.db.execute('PRAGMA page_size').fetchone()[0]; return {**dict(row),'database_bytes':page}
