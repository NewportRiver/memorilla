import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np, pytest
from lite.neural import NeuralConfig,NumpyAssociativeMemory
from lite.neural.attention import attention_agreement

def data(n=17):
 r=np.random.default_rng(7); return r.normal(size=384).astype('float32'),r.normal(size=(n,384)).astype('float32')
def test_shapes_determinism_and_parameter_bound():
 c=NeuralConfig(); a=NumpyAssociativeMemory(c); b=NumpyAssociativeMemory(c); q,m=data(); assert c.parameter_count()==a.parameter_count()==429312; assert np.array_equal(a.forward(q,m),b.forward(q,m)); assert a.forward(q,m).shape==(8,384)
def test_query_memory_sensitivity_and_permutation_invariance():
 q,m=data(); a=NumpyAssociativeMemory(); base=a.forward(q,m); assert not np.allclose(base,a.forward(-q,m)); assert not np.allclose(base,a.forward(q,m+0.1)); assert np.allclose(base,a.forward(q,m[::-1]),rtol=2e-5,atol=2e-5)
def test_mask_empty_and_invalid_inputs():
 q,m=data(); a=NumpyAssociativeMemory(); mask=np.ones(len(m),bool); mask[0]=False; assert np.allclose(a.forward(q,m,mask),a.forward(q,m[1:]),rtol=2e-5,atol=2e-5); empty=a.forward(q,np.empty((0,384),np.float32)); assert empty.shape==(8,384) and np.isfinite(empty).all();
 with pytest.raises(ValueError): a.forward(q,np.zeros((2,3),np.float32))
def test_streaming_full_agreement_and_stability():
 q,m=data(301); out=attention_agreement(NumpyAssociativeMemory(),q,m); assert out['allclose'] and out['max_abs_error']<2e-5
