"""Deterministic NumPy reference for query-conditioned fixed-size memory slots."""
from __future__ import annotations
import numpy as np

def _finite(x,name):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim==0 or not np.isfinite(x).all(): raise ValueError(f'{name} must be finite')
    return x

def stable_softmax(x,axis=-1):
    x=_finite(x,'logits'); m=np.max(x,axis=axis,keepdims=True); e=np.exp(x-m); z=np.sum(e,axis=axis,keepdims=True)
    if np.any(z==0) or not np.isfinite(z).all(): raise ValueError('softmax normalization failed')
    return e/z

def deterministic_weights(dim=384,slots=8,seed=20261119):
    if dim<=0 or slots<=0: raise ValueError('positive dimensions required')
    r=np.random.default_rng(seed)
    return {'slot_projection':(r.standard_normal((dim,slots*dim),dtype=np.float32)/np.sqrt(dim)).astype(np.float32),'slot_bias':np.zeros((slots,dim),np.float32),'key_projection':np.eye(dim,dtype=np.float32),'value_projection':np.eye(dim,dtype=np.float32)}

def _validate(memory,query,weights,dim,slots):
    m=_finite(memory,'memory'); q=_finite(query,'query')
    if m.ndim!=2 or m.shape[1]!=dim or m.shape[0]==0: raise ValueError('memory must be non-empty [records, dim]')
    if q.shape!=(dim,): raise ValueError('query must have shape [dim]')
    if not isinstance(slots,int) or slots<=0: raise ValueError('slots must be positive')
    return m,q

def initialize_slots(query,weights,dim=384,slots=8):
    q=_finite(query,'query')
    projected=q@weights['slot_projection']+weights['slot_bias'].reshape(-1)
    return np.tanh(projected.reshape(slots,dim)).astype(np.float32)

def attention(memory,query,weights=None,dim=384,slots=8,chunk_size=None):
    m,q=_validate(memory,query,weights or deterministic_weights(dim,slots),dim,slots); w=weights or deterministic_weights(dim,slots)
    s=initialize_slots(q,w,dim,slots); k=m@w['key_projection']; v=m@w['value_projection']; scale=np.float32(1/np.sqrt(dim))
    if chunk_size is None or chunk_size>=m.shape[0]:
        probs=stable_softmax((s@k.T)*scale,axis=1); return (probs@v).astype(np.float32)
    if chunk_size<=0: raise ValueError('chunk_size must be positive')
    logits_max=np.full((slots,),-np.inf,np.float32)
    for start in range(0,m.shape[0],chunk_size): logits_max=np.maximum(logits_max,np.max((s@k[start:start+chunk_size].T)*scale,axis=1))
    denom=np.zeros((slots,),np.float32); numer=np.zeros((slots,dim),np.float32)
    for start in range(0,m.shape[0],chunk_size):
        kk=k[start:start+chunk_size]; vv=v[start:start+chunk_size]; e=np.exp((s@kk.T)*scale-logits_max[:,None]); denom+=e.sum(axis=1); numer+=e@vv
    if np.any(denom==0) or not np.isfinite(denom).all(): raise ValueError('chunked normalization failed')
    return (numer/denom[:,None]).astype(np.float32)

def quantize_per_tensor(x):
    x=_finite(x,'values'); scale=max(float(np.max(np.abs(x)))/127.0, np.finfo(np.float32).tiny); q=np.clip(np.rint(x/scale),-127,127).astype(np.int8); return q,np.float32(scale)

def attention_stream(chunks,query,weights=None,dim=384,slots=8):
    """Two-pass global-normalized attention over a restartable chunk iterator."""
    w=weights or deterministic_weights(dim,slots); q=_finite(query,'query')
    s=initialize_slots(q,w,dim,slots); scale=np.float32(1/np.sqrt(dim)); maxes=np.full(slots,-np.inf,np.float32); seen=0
    for chunk in chunks:
        m=_finite(chunk,'memory');
        if m.ndim!=2 or m.shape[1]!=dim or m.shape[0]==0: raise ValueError('memory chunks must be non-empty [records, dim]')
        maxes=np.maximum(maxes,np.max((s@(m@w['key_projection']).T)*scale,axis=1)); seen+=m.shape[0]
    if seen==0: raise ValueError('memory stream is empty')
    denom=np.zeros(slots,np.float32); numer=np.zeros((slots,dim),np.float32)
    for chunk in chunks:
        m=_finite(chunk,'memory'); k=m@w['key_projection']; v=m@w['value_projection']; e=np.exp((s@k.T)*scale-maxes[:,None]); denom+=e.sum(axis=1); numer+=e@v
    if np.any(denom==0) or not np.isfinite(denom).all(): raise ValueError('stream normalization failed')
    return (numer/denom[:,None]).astype(np.float32)
