import numpy as np

from experiments.lite008f.projection import LinearEmbeddingBridge


def test_linear_bridge_contract_and_deterministic_initialization():
    a = LinearEmbeddingBridge(seed=19)
    b = LinearEmbeddingBridge(seed=19)
    assert a.weight.shape == (384, 576)
    assert a.bias.shape == (576,)
    assert np.array_equal(a.weight, b.weight)
    assert np.isfinite(a.forward(np.zeros((2, 384), np.float32))).all()


def test_linear_bridge_rejects_invalid_input():
    bridge = LinearEmbeddingBridge()
    with np.testing.assert_raises(ValueError):
        bridge.forward(np.zeros((1, 383), np.float32))
    bad = np.zeros((1, 384), np.float32)
    bad[0, 0] = np.nan
    with np.testing.assert_raises(ValueError):
        bridge.forward(bad)


def test_linear_bridge_checkpoint_round_trip(tmp_path):
    bridge = LinearEmbeddingBridge(seed=23)
    path = tmp_path / "bridge.npz"
    bridge.save(path, {"test": True})
    restored = LinearEmbeddingBridge.load(path)
    assert np.array_equal(bridge.weight, restored.weight)
    assert np.array_equal(bridge.bias, restored.bias)


def test_linear_bridge_sgd_reduces_alignment_loss():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(8, 384)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    target = rng.normal(size=(8, 576)).astype(np.float32)
    bridge = LinearEmbeddingBridge(seed=5)
    initial = float(np.mean((bridge.forward(x) - target) ** 2))
    bridge.train_sgd(x, target, steps=30, lr=0.05)
    final = float(np.mean((bridge.forward(x) - target) ** 2))
    assert final < initial
