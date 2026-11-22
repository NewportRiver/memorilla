# LITE-007O Scale-Normalized Gradient Attribution Report

## Baseline and reproducibility

Baseline verified:

- Repository branch: `feature/memorilla-lite-android`
- Commit: `75e9346`
- Working tree: clean at preflight
- LITE-007L checkpoint identities: verified against LITE-007N evidence and checkpoint manifest
- LITE-007N evidence identity: retained unchanged

Diagnostic command:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python benchmarks/lite/run_lite007o_diagnostics.py
```

The harness used the existing native gradient implementation, frozen initial/final associative checkpoints, and the same 360 eligible independent development queries/evidence. It did not write checkpoints or invoke an optimizer.

## Scale-normalized gradients

All 27 ranking-connected tensors were analyzed. For every tensor the output records shape, parameter count, raw gradient L2 statistics, gradient RMS per parameter, parameter RMS and L2 norm, gradient-to-parameter ratios, hypothetical learning-rate update ratios, component variance, and sample directional alignment.

Representative results:

| Tensor | Raw L2 mean | Gradient RMS | Parameter RMS | Grad/parameter RMS | Hypothetical update/parameter RMS |
|---|---:|---:|---:|---:|---:|
| `b0.q_w` | `1.79e-05` | `2.01e-07` | `0.08824` | `2.28e-06` | `2.28e-09` |
| `b0.k_w` | `3.16e-05` | `2.05e-07` | `0.05107` | `4.01e-06` | `4.01e-09` |
| `b0.v_w` | `0.07415` | `4.08e-04` | `0.05128` | `7.96e-03` | `7.96e-06` |
| `query_w` | `0.07722` | `4.20e-04` | `0.05101` | `8.23e-03` | `8.23e-06` |
| `score_slot_w` | `0.13704` | `1.95e-03` | `0.08886` | `2.19e-02` | `2.19e-05` |
| `score_mem_w` | `0.24398` | `1.95e-03` | `0.05117` | `3.81e-02` | `3.81e-05` |
| `slots` | `0.09600` | `4.17e-03` | `0.08759` | `4.76e-02` | `4.76e-05` |

The global gradient norm mean was `0.46770`; `9.72%` of examples exceeded the clipping norm of `1.0`. These are hypothetical update calculations only.

The important result is that attention block query/key attenuation survives parameter-RMS normalization. It is therefore not explained solely by tensor dimensionality or parameter scale. It remains a parameterization- and normalization-dependent attribution, not a complete causal proof.

## Jacobian and finite-difference attribution

Group-level native gradient RMS means:

- Query projection: `0.00463`
- Key projection: `0.00267`
- Value projection: `0.00455`
- Attention output: `0.00463`
- Latent slots: `0.00276`
- Final scoring: `0.00788`

Centered finite differences used copied in-memory states with `epsilon=1e-2`; checkpoint files remained untouched.

Validation was strongest for:

- Latent slots: mean relative error approximately `0.053`, 100% directional sign agreement.
- Final score projection: mean relative error approximately `0.072`, 100% directional sign agreement.

Upstream validation was weaker:

- `query_w`: mean relative error approximately `4.37`, 97.5% sign agreement.
- `mem_k_w`: approximately `0.67`, 100% sign agreement.
- `mem_v_w`: approximately `1.22`, 97.5% sign agreement.
- `b0.attn_out_w`: approximately `0.51`, 95% sign agreement.

This discrepancy is itself a limitation. The report treats native upstream gradient magnitudes as evidence of attenuation but does not claim that every upstream Jacobian measurement is independently validated. Correcting this attribution contract should precede any causal training pilot.

## Attention-axis and routing analysis

The implementation applies softmax over eligible evidence positions for each slot and attention head. The eight slots are not the softmax axis. Entropy is averaged over slot/head rows.

Observed final entropy:

- 8 eligible evidence positions: `2.079417`, approximately `ln(8)`
- 16 eligible evidence positions: `2.772562`, approximately `ln(16)`
- 32 eligible evidence positions: `3.465708`, approximately `ln(32)`

Thus the observed `2.77258` corresponds to near-uniform attention over 16 eligible evidence positions, not 16 slots.

Mean attention-logit standard deviation was approximately `0.00701` at the final checkpoint. Initial and final entropy/logit spread were effectively unchanged:

- Initial entropy mean: `2.772562`
- Final entropy mean: `2.772562`
- Initial logit spread: `0.007063`
- Final logit spread: `0.007010`

LITE-007N measured required-evidence enrichment of `0.99876` relative to uniform. Together these results support insufficient evidence-selective routing, while query sensitivity and nonzero gradients show that the model is not entirely query-insensitive.

## Scientific decision

The most likely measurable bottleneck is genuine attention-path attenuation coupled with near-uniform softmax routing. Parameter-scale normalization does not erase the signal. However, the evidence does not prove that changing attention temperature would improve retrieval, and upstream finite-difference validation is incomplete.

One corrective variable is scientifically defensible for a future authorized pilot: attention-logit temperature or equivalent attention-logit scale, with no other changes. That pilot is not authorized here.

Until the attribution discrepancy is repaired and an owner-approved one-variable pilot succeeds on the independent corpus, conventional retrieval remains preferable for the Android objective. No broad associative retraining is recommended.

## Resource and safety record

The diagnostic process used one numerical thread and one computational subprocess. Recorded peak RSS was below the 256 MiB ceiling. No GPU, cloud inference, new dependency, memory database, external provider, holdout, APEX integration, or LITE-008 implementation was used.

LITE-008 remains blocked pending owner review.
