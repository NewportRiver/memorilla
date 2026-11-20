from __future__ import annotations

import numpy as np


def sgd_step(state, gradients, learning_rate):
    if learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError('learning_rate must be finite and positive')
    for name, grad in gradients.items():
        if name not in state or not np.isfinite(grad).all():
            raise ValueError(f'invalid SGD gradient: {name}')
        state[name][...] = (state[name] - np.float32(learning_rate) * grad).astype(np.float32)


def snapshot_frozen(state, trainable_names):
    return {name: value.copy() for name, value in state.items() if name not in set(trainable_names)}


def verify_frozen(state, snapshot):
    return all(np.array_equal(state[name], value) for name, value in snapshot.items())