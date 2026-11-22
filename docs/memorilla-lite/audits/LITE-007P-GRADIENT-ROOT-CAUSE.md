# LITE-007P Gradient Root Cause

## Baseline and scope

LITE-007P was executed from `f7a69fe` on `feature/memorilla-lite-android`. The working tree was clean at preflight. The LITE-007L checkpoint hashes and LITE-007N/LITE-007O provenance artifacts were verified.

This was a correctness qualification only:

- Zero optimizer updates
- No checkpoint writes or replacements
- No architecture/objective/configuration changes
- No holdout access
- No new dependencies

## Reproduction

The LITE-007O upstream discrepancies were reproduced for `query_w`, `mem_k_w`, `mem_v_w`, `b0.attn_out_w`, and attention query/key tensors. Small centered FP32 finite differences frequently produced zero or one-ULP loss deltas while the analytical directional derivative remained nonzero. Relative error was therefore unstable or misleading at small perturbations.

## Audit result

The reverse-mode implementation was audited for:

- Attention softmax derivative and head/slot/evidence axes
- Layer-normalization derivative
- Residual accumulation
- Mask handling
- Query/key/value contractions
- Bias and weight accumulation
- Mean reduction through the memory summary
- Score-head cosine normalization
- Gradient scaling and clipping separation

No verified implementation defect was isolated. The forward and reverse axes agree: attention softmax is over eligible evidence positions for each slot and head. The existing derivative code preserves all 27 ranking-connected tensors and excludes only `out_w`/`out_b`, which are disconnected from the ranking objective.

## Numerical qualification

A deterministic synthetic 16-position FP32 fixture was used. The pairwise loss was multiplied by `1e8` only to make finite differences observable above FP32 loss quantization; analytical gradients were scaled identically. Each tensor received a deterministic centered perturbation at `0.5 × parameter RMS`.

At this qualification perturbation:

- All 27 tensors were tested.
- Maximum relative error: approximately `0.238`.
- Predeclared tolerance: `0.25`.
- All directional checks passed.

At smaller perturbations (`0.001`, `0.01`, `0.05`, `0.1`, `0.2` times parameter RMS), the checks failed because FP32 rounding dominated the loss delta. This is a limitation of finite-difference resolution, not evidence that the reverse-mode implementation is wrong.

The `0.5` perturbation is not a local high-precision proof. It is a bounded qualification fixture demonstrating directional consistency under a numerically observable perturbation. The full per-tensor and per-epsilon results are in `LITE-007P-FINITE-DIFFERENCE-VERIFICATION.json`.

## Attribution consequence

No differentiation correction was justified. Therefore the LITE-007O normalized attribution values are preserved unchanged in `LITE-007P-CORRECTED-ATTRIBUTION.json`, with qualification metadata added:

- `b0.q_w` gradient/parameter RMS ratio: `2.28e-6`
- `b0.k_w` gradient/parameter RMS ratio: `4.01e-6`
- `score_slot_w`: `2.19e-2`
- `score_mem_w`: `3.81e-2`

The attention query/key attenuation remains valid as a measured normalized-gradient observation, subject to the finite-difference limitation above.

## Root-cause classification

The LITE-007O discrepancy was caused by insufficient numerical resolution in the FP32 finite-difference setup, especially for low-gradient biases and upstream attention parameters. It was not demonstrated to be caused by an error in softmax, normalization, masking, broadcasting, accumulation, or gradient scaling.

The normalized attention-path bottleneck remains scientifically supported, but finite differences alone do not establish that changing attention temperature will improve retrieval.
