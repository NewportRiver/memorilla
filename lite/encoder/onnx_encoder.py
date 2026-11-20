"""Lazy ONNX Runtime wrapper for the pinned ARM64 MiniLM artifact."""
from __future__ import annotations
import hashlib,json,os,time,warnings
from pathlib import Path
import numpy as np
from .manifest import *
from .tokenizer import MiniLMTokenizer
from lite.runtime.telemetry import rss_bytes,thread_count
class MiniLMEncoder:
 def __init__(self,cache_dir=None,max_length=128,threads=2,providers=None):
  if threads>2 or threads<1: raise ValueError('threads must be 1 or 2')
  paths=model_paths(cache_dir); self.paths=paths; self.max_length=max_length; self.threads=threads
  if not paths['model'].is_file(): raise FileNotFoundError(paths['model'])
  got=hashlib.sha256(paths['model'].read_bytes()).hexdigest()
  if got!=MODEL_SHA256: raise ValueError(f'model checksum mismatch: {got}')
  self.tokenizer=MiniLMTokenizer(paths['vocab'],max_length=max_length,do_lower_case=True)
  import onnxruntime as ort
  opts=ort.SessionOptions(); opts.intra_op_num_threads=threads; opts.inter_op_num_threads=1; opts.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
  self.runtime_version=ort.__version__; self.providers=providers or ['CPUExecutionProvider']
  self.session=ort.InferenceSession(str(paths['model']),sess_options=opts,providers=self.providers)
  actual=set(self.session.get_providers());
  if 'CPUExecutionProvider' not in actual: raise RuntimeError('CPU provider unavailable')
  self.providers=actual
  names={x.name for x in self.session.get_inputs()}; required={'input_ids','attention_mask','token_type_ids'}
  if not required.issubset(names): raise RuntimeError(f'input contract mismatch: {names}')
 def identity(self): return {'repo':MODEL_REPO,'revision':MODEL_REVISION,'artifact':MODEL_ARTIFACT,'sha256':MODEL_SHA256,'license':MODEL_LICENSE,'dimension':OUTPUT_DIM,'max_input_tokens':self.max_length,'runtime':self.runtime_version,'providers':sorted(self.providers),'threads':self.threads,'pooling':'attention-mask mean then L2 normalization'}
 def encode(self,text): return self.encode_batch([text])[0]
 def encode_batch(self,texts):
  batch=self.tokenizer.batch(list(texts)); feeds={k:np.asarray(batch[k],dtype=np.int64) for k in ('input_ids','attention_mask','token_type_ids')}
  out=self.session.run(['last_hidden_state'],feeds)[0].astype(np.float32,copy=False); mask=feeds['attention_mask'].astype(np.float32)[...,None]; summed=(out*mask).sum(axis=1); denom=np.maximum(mask.sum(axis=1),1.0); emb=summed/denom; norms=np.linalg.norm(emb,axis=1,keepdims=True); emb=emb/np.maximum(norms,np.finfo(np.float32).tiny)
  if emb.shape[1]!=OUTPUT_DIM or not np.isfinite(emb).all(): raise RuntimeError('invalid embedding output')
  self.last_metadata={'input_tokens':[len(x) for x in batch['input_ids']],'truncated':batch['truncated'],'rss_bytes':rss_bytes(),'threads':thread_count()}
  return emb.astype(np.float32)
 def close(self): self.session=None; self.tokenizer=None
