import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.numpy_gradients import memory_projection_gradients
from lite.training.numpy_optimizer import sgd_step


def case(seed=7, records=4):
    rng = np.random.default_rng(seed)
    model = NumpyAssociativeMemory(NeuralConfig(seed=seed))
    query = rng.normal(size=384).astype(np.float32)
    memory = rng.normal(size=(records, 384)).astype(np.float32)
    valid = np.ones(records, dtype=bool)
    positive = np.zeros(records, dtype=bool)
    positive[0] = True
    positive[-1] = records > 2
    return model, query, memory, positive, valid


def test_memory_projection_gradients_finite_and_authorized():
    model, query, memory, positive, valid = case()
    loss, _, gradients, cache = memory_projection_gradients(model, query, memory, positive, valid)
    assert np.isfinite(loss)
    assert set(gradients) == {'mem_k_w', 'mem_k_b', 'mem_v_w', 'mem_v_b'}
    assert all(np.isfinite(x).all() for x in gradients.values())
    assert cache['latent'].shape == (8, 128)


def test_memory_projection_gradient_descent_reduces_diagnostic_loss():
    model, query, memory, positive, valid = case(11, 6)
    initial, _, gradients, _ = memory_projection_gradients(model, query, memory, positive, valid)
    sgd_step(model.state, gradients, 1e-4)
    final, _, _, _ = memory_projection_gradients(model, query, memory, positive, valid)
    assert final < initial


def test_masked_and_padded_records_do_not_change_projection_gradient():
    model, query, memory, positive, valid = case(13, 4)
    padded = np.vstack([memory, np.zeros((2, 384), np.float32)])
    padded_valid = np.concatenate([valid, [False, False]])
    padded_positive = np.concatenate([positive, [False, False]])
    a = memory_projection_gradients(model, query, memory, positive, valid)[2]
    b = memory_projection_gradients(model, query, padded, padded_positive, padded_valid)[2]
    assert all(np.allclose(a[k], b[k], rtol=1e-5, atol=1e-5) for k in a)
