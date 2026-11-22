import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.numpy_gradients import associative_gradients, pairwise_loss_and_score_gradient, score_head_forward


def _fixture(seed=31, n=16):
    rng = np.random.default_rng(seed)
    model = NumpyAssociativeMemory(NeuralConfig(seed=seed))
    q = rng.normal(size=384).astype('f4')
    mem = rng.normal(size=(n, 384)).astype('f4')
    positive = np.zeros(n, bool); positive[[1, 7]] = True
    return model, q, mem, positive, np.ones(n, bool)


def _loss(model, q, mem, positive, valid):
    _, latent = model.forward(q, mem, valid, return_latent=True)
    scores = score_head_forward(latent, mem, model.state, valid)[0]
    return float(pairwise_loss_and_score_gradient(scores, positive, valid)[0])


def _clone(model):
    return NumpyAssociativeMemory(model.config, state={k: v.copy() for k, v in model.state.items()})


def test_all_27_directional_gradients_match_scaled_centered_difference():
    model, q, mem, positive, valid = _fixture()
    loss, _, grads, _ = associative_gradients(model, q, mem, positive, valid)
    assert np.isfinite(loss)
    rng = np.random.default_rng(91)
    scale_loss = 100_000_000.0
    epsilon = 0.5
    errors = []
    for name in sorted(grads):
        direction = rng.normal(size=model.state[name].shape).astype('f4')
        direction /= np.float32(max(np.linalg.norm(direction), 1e-12))
        parameter_scale = max(float(np.sqrt(np.mean(model.state[name] ** 2))), 1e-3)
        delta = direction * np.float32(epsilon * parameter_scale)
        plus = _clone(model); minus = _clone(model)
        plus.state[name] += delta; minus.state[name] -= delta
        numerical = scale_loss * (_loss(plus, q, mem, positive, valid) - _loss(minus, q, mem, positive, valid)) / (2 * epsilon)
        analytical = scale_loss * float(np.sum(grads[name] * direction) * parameter_scale)
        denominator = max(abs(analytical), 1e-3)
        errors.append(abs(numerical - analytical) / denominator)
    assert len(errors) == 27
    assert max(errors) < 0.25
