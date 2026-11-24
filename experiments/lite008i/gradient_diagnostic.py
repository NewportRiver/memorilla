"""LITE-008I isolated differentiable answer-token diagnostic.

This intentionally uses a tiny synthetic frozen causal decoder, not SmolLM2.
It verifies the learning mechanism's mathematics without loading the blocked
full official model.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn as nn


class TinyFrozenCausalDecoder(nn.Module):
    def __init__(self, hidden: int = 8, vocab: int = 5) -> None:
        super().__init__()
        self.mix = nn.Linear(hidden, hidden, bias=False)
        self.out = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.mix.weight.copy_(torch.eye(hidden))
            self.out.weight.zero_()
            self.out.weight[2, 0] = 1.0
            self.out.weight[3, 1] = 1.0
        self.requires_grad_(False)
        self.eval()

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        # Causal prefix state: position t can depend only on positions <= t.
        states = torch.cumsum(self.mix(inputs_embeds), dim=1)
        return self.out(states)


def run() -> dict:
    torch.manual_seed(19)
    decoder = TinyFrozenCausalDecoder()
    fixed = torch.zeros(1, 3, 8)
    fixed[:, 0, 0] = 0.2
    fixed[:, 2, 1] = 0.1
    memory = torch.zeros(1, 8, requires_grad=True)
    target = torch.tensor([2])
    # The answer label supervises only the final causal position.
    def loss_for(mem: torch.Tensor) -> torch.Tensor:
        embeds = fixed.clone()
        embeds[:, 1, :] = mem
        logits = decoder(embeds)
        return nn.functional.cross_entropy(logits[:, 2, :], target)
    before = float(loss_for(memory).detach())
    loss = loss_for(memory)
    grad = torch.autograd.grad(loss, memory, create_graph=False)[0]
    finite = bool(torch.isfinite(grad).all())
    analytic = float(grad[0, 0])
    epsilon = 1e-3
    plus = memory.detach().clone(); plus[0, 0] += epsilon
    minus = memory.detach().clone(); minus[0, 0] -= epsilon
    numerical = float((loss_for(plus) - loss_for(minus)) / (2 * epsilon))
    relative_error = abs(analytic - numerical) / max(abs(analytic), 1e-8)
    optimizer = torch.optim.SGD([memory], lr=1.0)
    for _ in range(10):
        optimizer.zero_grad(); current = loss_for(memory); current.backward(); optimizer.step()
    after = float(loss_for(memory).detach())
    frozen_grads = [p.grad for p in decoder.parameters()]
    return {
        "synthetic_only": True,
        "decoder_parameter_count": sum(p.numel() for p in decoder.parameters()),
        "decoder_parameters_frozen": all(not p.requires_grad for p in decoder.parameters()),
        "decoder_parameter_grads_absent": all(g is None for g in frozen_grads),
        "external_embedding_gradient_finite": finite,
        "answer_only_position": 2,
        "ignored_positions": [0, 1],
        "analytic_gradient": analytic,
        "numerical_gradient": numerical,
        "relative_error": relative_error,
        "loss_before": before,
        "loss_after": after,
        "embedding_optimization_reduced_loss": after < before,
    }


if __name__ == "__main__":
    result = run()
    out = Path(__file__).resolve().parents[2] / "docs/memorilla-lite/audits/LITE-008I-GRADIENT-VERIFICATION.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
