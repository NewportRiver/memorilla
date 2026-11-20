from __future__ import annotations
import numpy as np
from .numpy_model import NumpyAssociativeMemory

def full_attention(model,query,memory,mask=None): return model.forward(query,memory,mask,return_attention=True)
def streamed_attention(model,query,chunks): return model.forward_stream(query,chunks)
def attention_agreement(model,query,memory,mask=None,rtol=2e-5,atol=2e-5):
 full=model.forward(query,memory,mask); chunks=[]; step=128
 for i in range(0,len(memory),step): chunks.append((memory[i:i+step],None if mask is None else mask[i:i+step]))
 stream=model.forward_stream(query,chunks); delta=float(np.max(np.abs(full-stream))) if full.size else 0.0
 return {'max_abs_error':delta,'allclose':bool(np.allclose(full,stream,rtol=rtol,atol=atol)),'full_shape':list(full.shape),'stream_shape':list(stream.shape)}
