import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.numpy_gradients import pairwise_loss_and_score_gradient, scoring_head_gradients
from lite.training.numpy_optimizer import sgd_step


def test_positive_score_increase_reduces_loss_and_negative_increase_raises_it():
    scores = np.array([0.2, -0.1, 0.0], np.float32)
    positive = np.array([True, False, False])
    valid = np.ones(3, bool)
    base, _ = pairwise_loss_and_score_gradient(scores, positive, valid)
    higher_positive, _ = pairwise_loss_and_score_gradient(scores + np.array([0.1, 0, 0], np.float32), positive, valid)
    higher_negative, _ = pairwise_loss_and_score_gradient(scores + np.array([0, 0.1, 0], np.float32), positive, valid)
    assert higher_positive < base < higher_negative


def test_small_gradient_descent_step_reduces_deterministic_fixture_loss():
    model = NumpyAssociativeMemory(NeuralConfig())
    rng = np.random.default_rng(31)
    latent = rng.normal(size=(8, 128)).astype(np.float32)
    memory = rng.normal(size=(5, 384)).astype(np.float32)
    positive = np.array([True, False, True, False, False])
    valid = np.ones(5, bool)
    initial, _, gradients = scoring_head_gradients(latent, memory, model.state, positive, valid)
    sgd_step(model.state, gradients, 0.0001)
    final, _, _ = scoring_head_gradients(latent, memory, model.state, positive, valid)
    assert final < initial


def test_multiple_scoring_parameters_have_finite_difference_agreement():
    model = NumpyAssociativeMemory(NeuralConfig())
    rng = np.random.default_rng(32)
    latent = rng.normal(size=(8, 128)).astype(np.float32)
    memory = rng.normal(size=(5, 384)).astype(np.float32)
    positive = np.array([True, False, True, False, False])
    valid = np.ones(5, bool)
    _, _, gradients = scoring_head_gradients(latent, memory, model.state, positive, valid)
    for name, index in [('score_slot_w', (0, 0)), ('score_slot_b', (0,)), ('score_mem_w', (0, 0)), ('score_mem_b', (0,))]:
        old = model.state[name][index]
        epsilon = 1e-3
        model.state[name][index] = old + epsilon
        plus = scoring_head_gradients(latent, memory, model.state, positive, valid)[0]
        model.state[name][index] = old - epsilon
        minus = scoring_head_gradients(latent, memory, model.state, positive, valid)[0]
        model.state[name][index] = old
        finite_difference = (plus - minus) / (2 * epsilon)
        assert abs(float(finite_difference) - float(gradients[name][index])) < 3e-3
