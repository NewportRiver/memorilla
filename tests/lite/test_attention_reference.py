import numpy as np,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
from lite.attention.reference import attention,deterministic_weights,quantize_per_tensor,stable_softmax
D=384; S=8
def data(n,seed):
 r=np.random.default_rng(seed); return r.standard_normal((n,D),dtype=np.float32),r.standard_normal(D,dtype=np.float32)

def test_fixed_shape_and_chunk_consistency():
 w=deterministic_weights(D,S)
 for n in (100,1000,5000):
  m,q=data(n,n); a=attention(m,q,w,D,S); b=attention(m,q,w,D,S,257); assert a.shape==(S,D); assert np.max(np.abs(a-b))<2e-5

def test_reproducible_query_and_corpus_dependence():
 w=deterministic_weights(D,S); m,q=data(100,1); a=attention(m,q,w,D,S); assert np.array_equal(a,attention(m,q,w,D,S)); assert np.max(np.abs(a-attention(m,q+0.01,w,D,S)))>1e-7; m2=m.copy(); m2[0,0]+=1; assert np.max(np.abs(a-attention(m2,q,w,D,S)))>1e-8

def test_softmax_stability_and_invalid_inputs():
 p=stable_softmax(np.array([[10000,-10000]],np.float32)); assert np.isfinite(p).all() and np.allclose(p.sum(1),1); w=deterministic_weights(D,S); m,q=data(3,2)
 with pytest.raises(ValueError): attention(np.empty((0,D),np.float32),q,w,D,S)
 with pytest.raises(ValueError): attention(np.full((3,D),np.nan,np.float32),q,w,D,S)
 with pytest.raises(ValueError): attention(m,np.full(D,np.inf,np.float32),w,D,S)

def test_quantization_error_is_measured():
 m,q=data(100,3); z=attention(m,q,deterministic_weights(D,S),D,S); qi,scale=quantize_per_tensor(z); restored=qi.astype(np.float32)*scale; assert qi.dtype==np.int8; assert float(np.max(np.abs(z-restored)))<=float(scale)
