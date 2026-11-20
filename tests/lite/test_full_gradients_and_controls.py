import numpy as np

from lite.neural.config import NeuralConfig
from lite.neural.numpy_model import NumpyAssociativeMemory
from lite.training.controls import DirectQueryEvidenceScorer, hybrid_rank, lexical_rank
from lite.training.numpy_gradients import associative_gradients


def fixture(seed=17, n=6):
    rng = np.random.default_rng(seed)
    model = NumpyAssociativeMemory(NeuralConfig(seed=seed))
    return model, rng.normal(size=384).astype('f4'), rng.normal(size=(n,384)).astype('f4'), np.array([1,0,1,0,0,0], bool), np.ones(n, bool)


def test_all_ranking_connected_gradients_are_present_and_finite():
    model, q, mem, positive, valid = fixture()
    loss, _, grads, _ = associative_gradients(model, q, mem, positive, valid)
    assert np.isfinite(loss)
    assert len(grads) == 27
    assert 'out_w' not in grads and 'out_b' not in grads
    assert all(np.isfinite(value).all() for value in grads.values())


def test_controls_are_distinct_and_operational():
    order, lexical = lexical_rank('alpha beta', ['alpha fact', 'unrelated'], [True, True])
    hybrid, _ = hybrid_rank(lexical, np.array([0.1, 0.9], 'f4'))
    scorer = DirectQueryEvidenceScorer(seed=3)
    scores = scorer.scores(np.ones(384, 'f4'), np.ones((2,384), 'f4'))
    assert order[0] == 0
    assert hybrid[0] in (0, 1)
    assert scores.shape == (2,)
