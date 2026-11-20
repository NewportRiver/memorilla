from __future__ import annotations

import numpy as np

_TRAINABLE = ('score_slot_w', 'score_slot_b', 'score_mem_w', 'score_mem_b')


def _finite(x, name):
    x = np.asarray(x, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f'{name} contains non-finite values')
    return x


def _normalize(x, axis=-1):
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError('scoring projection has zero or non-finite norm')
    return x / norm


def score_head_forward(latent, memory, state, valid_mask):
    """Return slot/evidence cosine scores using only the existing score head."""
    latent = _finite(latent, 'latent')
    memory = _finite(memory, 'memory')
    valid = np.asarray(valid_mask, dtype=bool)
    if latent.ndim != 2 or memory.ndim != 2 or valid.shape != (memory.shape[0],):
        raise ValueError('invalid scoring-head shapes')
    a = _normalize(latent @ state['score_slot_w'] + state['score_slot_b'])
    raw_b = memory @ state['score_mem_w'] + state['score_mem_b']
    b = np.zeros_like(raw_b, dtype=np.float32)
    if np.any(valid):
        b[valid] = _normalize(raw_b[valid])
    similarities = a @ b.T
    scores = np.full((memory.shape[0],), -np.inf, dtype=np.float32)
    slots = np.full((memory.shape[0],), -1, dtype=np.int64)
    for i in range(memory.shape[0]):
        if valid[i]:
            slots[i] = int(np.argmax(similarities[:, i]))
            scores[i] = similarities[slots[i], i]
    return scores, slots, (a, b, similarities)


def pairwise_loss_and_score_gradient(scores, positive_mask, valid_mask):
    """Multi-positive logistic ranking loss and d(loss)/d(scores)."""
    scores = _finite(np.where(np.isfinite(scores), scores, 0.0), 'scores')
    positive = np.asarray(positive_mask, dtype=bool)
    valid = np.asarray(valid_mask, dtype=bool)
    if scores.shape != positive.shape or scores.shape != valid.shape:
        raise ValueError('ranking mask shape mismatch')
    pos = np.flatnonzero(positive & valid)
    neg = np.flatnonzero(~positive & valid)
    grad = np.zeros_like(scores, dtype=np.float32)
    if not len(pos) or not len(neg):
        return np.float32(0.0), grad
    diffs = scores[neg, None] - scores[None, pos]
    loss = np.logaddexp(0.0, diffs).mean()
    sigmoid = 1.0 / (1.0 + np.exp(-diffs))
    grad[pos] -= sigmoid.sum(axis=0) / (len(pos) * len(neg))
    grad[neg] += sigmoid.sum(axis=1) / (len(pos) * len(neg))
    return np.float32(loss), grad


def scoring_head_gradients(latent, memory, state, positive_mask, valid_mask):
    """Compute exact Float32 gradients for the four existing scoring-head tensors."""
    scores, chosen, cache = score_head_forward(latent, memory, state, valid_mask)
    loss, dscore = pairwise_loss_and_score_gradient(scores, positive_mask, valid_mask)
    a, b, similarities = cache
    da = np.zeros_like(a, dtype=np.float64)
    db = np.zeros_like(b, dtype=np.float64)
    for i, slot in enumerate(chosen):
        if slot < 0 or dscore[i] == 0.0:
            continue
        g = float(dscore[i])
        av = a[slot].astype(np.float64)
        bv = b[i].astype(np.float64)
        da[slot] += g * (bv - similarities[slot, i] * av)
        db[i] += g * (av - similarities[slot, i] * bv)
    slot_raw = latent @ state['score_slot_w'] + state['score_slot_b']
    mem_raw = memory @ state['score_mem_w'] + state['score_mem_b']
    slot_norm = np.linalg.norm(slot_raw, axis=1, keepdims=True)
    mem_norm = np.linalg.norm(mem_raw, axis=1, keepdims=True)
    mem_norm = np.maximum(mem_norm, 1e-12)
    da_raw = (da - a.astype(np.float64) * (da * a.astype(np.float64)).sum(axis=1, keepdims=True)) / slot_norm
    db_raw = (db - b.astype(np.float64) * (db * b.astype(np.float64)).sum(axis=1, keepdims=True)) / mem_norm
    grads = {
        'score_slot_w': (latent.astype(np.float64).T @ da_raw).astype(np.float32),
        'score_slot_b': da_raw.sum(axis=0).astype(np.float32),
        'score_mem_w': (memory.astype(np.float64).T @ db_raw).astype(np.float32),
        'score_mem_b': db_raw.sum(axis=0).astype(np.float32),
    }
    if not all(np.isfinite(x).all() for x in grads.values()):
        raise FloatingPointError('non-finite scoring-head gradient')
    return loss, scores, grads


def clip_gradients(grads, max_norm):
    if max_norm <= 0:
        raise ValueError('max_norm must be positive')
    norm = float(np.sqrt(sum(np.square(x.astype(np.float64)).sum() for x in grads.values())))
    scale = min(1.0, max_norm / norm) if norm else 1.0
    return {k: (v * scale).astype(np.float32) for k, v in grads.items()}, norm, scale


def trainable_names():
    return _TRAINABLE