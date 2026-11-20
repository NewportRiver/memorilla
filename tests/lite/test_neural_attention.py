import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from lite.neural import NumpyAssociativeMemory
from lite.neural.attention import full_attention,streamed_attention

def test_chunked_online_softmax_matches_full():
 r=np.random.default_rng(9); q=r.normal(size=384).astype('float32'); m=r.normal(size=(513,384)).astype('float32'); mask=np.ones(513,bool); mask[::7]=False; model=NumpyAssociativeMemory(); a=full_attention(model,q,m,mask)[0]; chunks=[(m[i:i+64],mask[i:i+64]) for i in range(0,len(m),64)]; b=streamed_attention(model,q,chunks); assert np.allclose(a,b,rtol=2e-5,atol=2e-5)
def test_empty_stream_is_finite():
 model=NumpyAssociativeMemory(); out=model.forward_stream(np.zeros(384,np.float32),[]); assert out.shape==(8,384) and np.isfinite(out).all()
