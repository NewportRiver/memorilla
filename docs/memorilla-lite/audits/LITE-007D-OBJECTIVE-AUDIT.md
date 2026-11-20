# LITE-007D Objective Audit

Status: corrected objective verified; pilot results from the earlier incorrect-sign implementation are explicitly limited.

## Objective sign

The authoritative ranking objective is:

`mean softplus(margin + negative_score - positive_score)`

Increasing a positive score lowers the loss. Increasing a negative score raises the loss. The PyTorch implementation in `lite/training/objectives.py` and the native NumPy implementation in `lite/training/numpy_gradients.py` now use the same sign.

## Directional verification

The deterministic objective-direction tests passed:

- Positive-score increase reduced loss.
- Negative-score increase increased loss.
- A small correct SGD step reduced loss on a fixed nonstationary fixture.
- Finite differences across `score_slot_w`, `score_slot_b`, `score_mem_w`, and `score_mem_b` matched analytical gradients within the configured tolerance.
- Targeted tests: 7 passed in 0.27 seconds.

## Historical impact

The LITE-007 preparation and LITE-007C gradient audit occurred before the shared PyTorch sign correction. They performed structural gradient/connectivity checks and no optimizer updates, so no trained weights or checkpoints are invalidated. However, any historical loss values, gradient directions, or hypothetical optimization interpretations based on the pre-correction objective are invalid for ranking-learning claims.

The following remain valid as structural evidence only:

- model forward compatibility;
- parameter naming and count;
- finite backward execution;
- memory-projection gradient connectivity;
- zero gradients for output projections outside the ranking path.

No prior LITE-007 optimizer result exists to invalidate because recorded optimizer updates were zero.

## Current status

The corrected objective is the only accepted objective for future training, evaluation, and checkpoint metadata. LITE-007 remains incomplete.
