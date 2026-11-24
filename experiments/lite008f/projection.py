"""Standalone deterministic 384 -> 576 decoder-embedding alignment bridge.

This module is deliberately independent of Memorilla's neural-memory state and
persistent checkpoints. It optimizes only a linear projection against frozen
reference decoder input embeddings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


class LinearEmbeddingBridge:
    def __init__(self, input_dim: int = 384, output_dim: int = 576, seed: int = 20261124):
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("dimensions must be positive")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.weight = (rng.normal(0.0, 0.02, (input_dim, output_dim))).astype(np.float32)
        self.bias = np.zeros((output_dim,), dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.input_dim or not np.isfinite(x).all():
            raise ValueError(f"input must be finite [batch,{self.input_dim}]")
        y = x @ self.weight + self.bias
        if not np.isfinite(y).all():
            raise FloatingPointError("bridge output is non-finite")
        return y.astype(np.float32, copy=False)

    def train_sgd(self, x: np.ndarray, target: np.ndarray, steps: int = 200, lr: float = 0.05) -> list[float]:
        x = np.asarray(x, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        if target.shape != (x.shape[0], self.output_dim) or not np.isfinite(target).all():
            raise ValueError("target shape or values invalid")
        if steps <= 0 or lr <= 0:
            raise ValueError("steps and lr must be positive")
        history: list[float] = []
        n = float(x.shape[0])
        for _ in range(steps):
            pred = self.forward(x)
            err = pred - target
            loss = float(np.mean(err * err))
            grad_w = (x.astype(np.float64).T @ err.astype(np.float64) / n).astype(np.float32)
            grad_b = err.mean(axis=0).astype(np.float32)
            self.weight = (self.weight - lr * grad_w).astype(np.float32)
            self.bias = (self.bias - lr * grad_b).astype(np.float32)
            history.append(loss)
        return history

    def state(self) -> dict[str, np.ndarray]:
        return {"weight": self.weight.copy(), "bias": self.bias.copy()}

    def save(self, path: str | Path, metadata: dict) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise ValueError("cannot save non-finite bridge")
        np.savez(path, weight=self.weight, bias=self.bias)
        payload = {
            "schema": "lite008f-linear-bridge-v1",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "parameter_count": int(self.weight.size + self.bias.size),
            "dtype": "float32",
            "seed": self.seed,
            "weight_sha256": hashlib.sha256(self.weight.tobytes()).hexdigest(),
            "bias_sha256": hashlib.sha256(self.bias.tobytes()).hexdigest(),
            **metadata,
        }
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return payload

    @classmethod
    def load(cls, path: str | Path) -> "LinearEmbeddingBridge":
        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as z:
            w = np.asarray(z["weight"], dtype=np.float32)
            b = np.asarray(z["bias"], dtype=np.float32)
        out = cls(meta["input_dim"], meta["output_dim"], meta["seed"])
        if w.shape != out.weight.shape or b.shape != out.bias.shape:
            raise ValueError("bridge checkpoint shape mismatch")
        if hashlib.sha256(w.tobytes()).hexdigest() != meta["weight_sha256"]:
            raise ValueError("bridge weight checksum mismatch")
        if hashlib.sha256(b.tobytes()).hexdigest() != meta["bias_sha256"]:
            raise ValueError("bridge bias checksum mismatch")
        out.weight, out.bias = w, b
        return out
