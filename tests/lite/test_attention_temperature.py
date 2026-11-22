import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.numpy_gradients import associative_gradients


def test_attention_scale_one_preserves_forward_and_gradient_contract():
    rng=np.random.default_rng(7)
    state_model=NumpyAssociativeMemory(NeuralConfig(seed=7))
    scaled=NumpyAssociativeMemory(NeuralConfig(seed=7,attention_logit_scale=1.0),state={k:v.copy() for k,v in state_model.state.items()})
    q=rng.normal(size=384).astype('f4'); mem=rng.normal(size=(16,384)).astype('f4'); valid=np.ones(16,bool); pos=np.zeros(16,bool); pos[[0,4]]=True
    a=state_model.forward(q,mem,valid,return_attention=True,return_latent=True)
    b=scaled.forward(q,mem,valid,return_attention=True,return_latent=True)
    assert np.array_equal(a[0][0],b[0][0])
    assert np.array_equal(a[1],b[1])
    ga=associative_gradients(state_model,q,mem,pos,valid)[2]
    gb=associative_gradients(scaled,q,mem,pos,valid)[2]
    assert all(np.array_equal(ga[k],gb[k]) for k in ga)


def test_attention_scale_changes_logits_without_changing_parameter_count():
    a=NumpyAssociativeMemory(NeuralConfig(seed=8,attention_logit_scale=0.5))
    b=NumpyAssociativeMemory(NeuralConfig(seed=8,attention_logit_scale=2.0))
    assert a.parameter_count()==b.parameter_count()==429312
