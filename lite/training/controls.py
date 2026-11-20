from __future__ import annotations

import numpy as np

from .numpy_gradients import pairwise_loss_and_score_gradient


def _tokens(text):
    return {x for x in ''.join(c.lower() if c.isalnum() else ' ' for c in text).split() if len(x) > 2}


def lexical_rank(query, texts, valid=None):
    q = _tokens(query); valid = np.ones(len(texts), bool) if valid is None else np.asarray(valid, bool)
    scores = np.asarray([len(q & _tokens(t)) / max(len(q | _tokens(t)), 1) if valid[i] else -np.inf for i, t in enumerate(texts)], np.float32)
    return np.argsort(-scores, kind='stable').tolist(), scores


def hybrid_rank(lexical_scores, semantic_scores, valid=None, rrf_k=60):
    valid = np.ones(len(lexical_scores), bool) if valid is None else np.asarray(valid, bool)
    lr = np.argsort(-np.asarray(lexical_scores), kind='stable'); sr = np.argsort(-np.asarray(semantic_scores), kind='stable')
    score = np.full(len(lr), -np.inf, np.float32)
    for rank, i in enumerate(lr):
        if valid[i]: score[i] = 1.0 / (rrf_k + rank + 1)
    for rank, i in enumerate(sr):
        if valid[i]: score[i] = (0.0 if not np.isfinite(score[i]) else score[i]) + 1.0 / (rrf_k + rank + 1)
    return np.argsort(-score, kind='stable').tolist(), score


class DirectQueryEvidenceScorer:
    """Separately trainable bilinear query/evidence score control."""
    def __init__(self, dimension=384, seed=20261120):
        rng = np.random.default_rng(seed)
        self.weight = (rng.standard_normal((dimension, dimension), dtype=np.float32) / np.sqrt(dimension)).astype(np.float32)

    def scores(self, query, memory, valid=None):
        valid = np.ones(len(memory), bool) if valid is None else np.asarray(valid, bool)
        q = query @ self.weight
        raw = memory @ q
        return np.where(valid, raw, -np.inf).astype(np.float32)

    def loss(self, query, memory, positive, valid):
        scores = self.scores(query, memory, valid)
        return pairwise_loss_and_score_gradient(scores, positive, valid)[0]


def query_independent_scores(model, memory, valid):
    zero_query = np.zeros(model.config.input_dim, np.float32)
    _, latent = model.forward(zero_query, memory, valid, return_latent=True)
    from .numpy_gradients import score_head_forward
    return score_head_forward(latent, memory, model.state, valid)[0]
