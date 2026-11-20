from __future__ import annotations
import hashlib,struct
import numpy as np
FORMATS={'float32':(4,'<f'),'float16':(2,None),'int8':(1,None)}
def _valid(x,dim=None):
 a=np.asarray(x,dtype=np.float32)
 if a.ndim!=1 or (dim is not None and a.size!=dim) or not np.isfinite(a).all(): raise ValueError('vector must be finite 1-D with expected dimension')
 return a
def encode_vector(vector,dtype='float32',dimension=None):
 a=_valid(vector,dimension)
 if dtype=='float32': raw=np.asarray(a,dtype='<f4').tobytes(); scale=None
 elif dtype=='float16': raw=np.asarray(a,dtype='<f2').tobytes(); scale=None
 elif dtype=='int8':
  maxabs=float(np.max(np.abs(a))); scale=max(maxabs/127.0,np.finfo(np.float32).tiny); raw=np.clip(np.rint(a/scale),-127,127).astype(np.int8).tobytes()
 else: raise ValueError('unsupported vector dtype')
 return raw,scale,hashlib.sha256(raw).hexdigest()
def decode_vector(raw,dtype,dimension,scale=None,checksum=None):
 if checksum and hashlib.sha256(raw).hexdigest()!=checksum: raise ValueError('vector checksum mismatch')
 width=FORMATS.get(dtype,(0,None))[0]
 if width==0 or len(raw)!=dimension*width: raise ValueError('malformed vector length')
 if dtype=='float32': out=np.frombuffer(raw,dtype='<f4').astype(np.float32)
 elif dtype=='float16': out=np.frombuffer(raw,dtype='<f2').astype(np.float32)
 else:
  if scale is None or not np.isfinite(scale) or scale<=0: raise ValueError('invalid int8 scale')
  out=np.frombuffer(raw,dtype=np.int8).astype(np.float32)*np.float32(scale)
 if not np.isfinite(out).all(): raise ValueError('decoded vector is non-finite')
 return out
def cosine(a,b):
 a=_valid(a); b=_valid(b,a.size); na=np.linalg.norm(a); nb=np.linalg.norm(b)
 if na==0 or nb==0:return 0.0
 return float(np.dot(a,b)/(na*nb))
