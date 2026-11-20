import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.numpy_gradients import (
    clip_gradients,
    pairwise_loss_and_score_gradient,
    scoring_head_gradients,
)
from lite.training.numpy_optimizer import sgd_step, snapshot_frozen, verify_frozen


def fixture():
    config = NeuralConfig()
    model = NumpyAssociativeMemory(config)
    rng = np.random.default_rng(23)
    latent = rng.normal(size=(config.slots, config.latent_dim)).astype(np.float32)
    memory = rng.normal(size=(5, config.input_dim)).astype(np.float32)
    return model, latent, memory, np.array([True, False, True, False, False]), np.ones(5, bool)


def test_scoring_head_gradients_are_finite_and_frozen_state_is_preserved():
    model, latent, memory, positive, valid = fixture()
    _, _, gradients = scoring_head_gradients(latent, memory, model.state, positive, valid)
    frozen = snapshot_frozen(model.state, gradients)
    clipped, norm, scale = clip_gradients(gradients, 1.0)
    before = model.state['score_slot_w'].copy()
    sgd_step(model.state, clipped, 0.001)
    assert norm > 0 and 0 < scale < 1
    assert not np.array_equal(before, model.state['score_slot_w'])
    assert verify_frozen(model.state, frozen)


def test_empty_positive_set_is_zero_and_finite():
    loss, gradient = pairwise_loss_and_score_gradient(
        np.zeros(5, np.float32), np.zeros(5, bool), np.ones(5, bool)
    )
    assert loss == 0.0 and np.isfinite(gradient).all() and not gradient.any()


def test_nonfinite_inputs_are_rejected():
    model, latent, memory, positive, valid = fixture()
    latent[0, 0] = np.nan
    try:
        scoring_head_gradients(latent, memory, model.state, positive, valid)
    except ValueError:
        return
    raise AssertionError('non-finite latent input was accepted')


def test_deterministic_fixture_gradients():
    a = fixture()
    b = fixture()
    ga = scoring_head_gradients(a[1], a[2], a[0].state, a[3], a[4])[2]
    gb = scoring_head_gradients(b[1], b[2], b[0].state, b[3], b[4])[2]
    assert all(np.array_equal(ga[name], gb[name]) for name in ga)
