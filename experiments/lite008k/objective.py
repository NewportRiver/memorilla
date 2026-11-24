"""Stable Float32 answer-NLL plus contrastive margin objective helpers."""
from __future__ import annotations
import torch

def answer_contrastive_loss(logits: torch.Tensor, correct_id: int, alternate_id: int, coefficient: float = 0.5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total loss, answer NLL, and correct-minus-alternate log-prob margin.

    Logits are explicitly promoted to Float32 before log-softmax. This helper is
    diagnostic-only and does not update any decoder or production memory state.
    """
    if logits.ndim != 2 or not logits.is_floating_point():
        raise ValueError('logits must be [batch,vocab] floating point')
    if coefficient < 0:
        raise ValueError('coefficient must be nonnegative')
    log_probs = logits.float().log_softmax(dim=-1)
    correct = log_probs[:, correct_id]
    alternate = log_probs[:, alternate_id]
    margin = correct - alternate
    nll = -correct.mean()
    loss = nll + coefficient * torch.nn.functional.softplus(-margin).mean()
    return loss, nll, margin.mean()
