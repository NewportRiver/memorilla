"""Pure scoring contracts used by LITE-008L audits."""
from __future__ import annotations

def answer_margin(answer_logprob: float, alternate_logprob: float) -> float:
    return float(answer_logprob-alternate_logprob)

def counterfactual_success(a_minus_b_margin: float) -> bool:
    return float(a_minus_b_margin)<0.0

def answer_positions(prompt_length: int, answer_token_count: int) -> tuple[int,int]:
    if prompt_length<1 or answer_token_count<1: raise ValueError('non-empty prompt and answer required')
    return prompt_length-1,prompt_length+answer_token_count-2
