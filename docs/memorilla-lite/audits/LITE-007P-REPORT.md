# LITE-007P Report

## Result

LITE-007P qualified the existing native associative reverse-mode differentiation implementation without changing it. The LITE-007O upstream discrepancy was reproduced and attributed to FP32 finite-difference conditioning rather than a verified derivative defect.

Baseline:

- Branch: `feature/memorilla-lite-android`
- Starting commit: `f7a69fe`
- Final commit: recorded after verification
- Existing LITE-007L checkpoints: unchanged
- LITE-007N/LITE-007O evidence identities: preserved

## Reproducible commands

Diagnostic qualification:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python benchmarks/lite/run_lite007p_qualification.py
```

Targeted deterministic gradient test:

```sh
PYTHONPATH=. python -m pytest -q tests/lite/test_gradient_integrity.py
```

Complete LITE regression:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m pytest -q --confcutdir=tests/lite tests/lite
```

## Differentiation audit

The audit covered softmax derivatives, attention axes, layer normalization, residual paths, masks, broadcast/reduction behavior, score-head normalization, and all ranking-connected tensors.

No verified reverse-mode defect was found. The implementation returns 27 finite gradients and excludes only the disconnected output projection tensors `out_w` and `out_b`.

The apparent discrepancy came from numerical finite differences. At small FP32 perturbations, low-gradient upstream and bias directions produced loss changes at or below Float32 resolution. Increasing loss scale alone did not solve the issue; a centered perturbation of `0.5 × parameter RMS` was required for a stable qualification fixture.

## Finite-difference qualification

The deterministic 16-position fixture tested all 27 tensors at epsilon values `0.001`, `0.01`, `0.05`, `0.1`, `0.2`, and `0.5` times parameter RMS.

- Epsilon values through `0.2` were explicitly classified as quantization-limited.
- Epsilon `0.5` passed the declared maximum relative-error threshold of `0.25` for all 27 tensors.
- Maximum relative error at the qualifying epsilon: approximately `0.238`.
- The loss was scaled by `1e8` equally for analytical and numerical paths only to make FP32 deltas observable.
- The qualifying perturbation is nonlocal; it is not presented as a high-precision infinitesimal derivative proof.

Full per-tensor results are in `LITE-007P-FINITE-DIFFERENCE-VERIFICATION.json`.

## Corrected attribution

No source correction was warranted. `LITE-007P-CORRECTED-ATTRIBUTION.json` preserves LITE-007O values and adds the qualification status.

The normalized attention-path signal remains:

- `b0.q_w` gradient/parameter RMS: `2.28e-6`
- `b0.k_w`: `4.01e-6`
- `score_slot_w`: `2.19e-2`
- `score_mem_w`: `3.81e-2`

Thus the attention query/key attenuation remains supported after normalization, but it is not claimed to be the sole causal explanation for retrieval failure.

## Regression and resource record

- Complete LITE regression: `82 passed`, `0 failed`, `0 skipped`
- One known Android ONNX Runtime platform warning
- New deterministic 27-tensor gradient qualification: passed
- Python compilation and diff checks: passed
- One initial regression attempt encountered a live available-RAM preflight failure; no limits were relaxed and the immediate rerun passed.

No optimizer updates occurred. No checkpoints were modified. No final holdout or external inference was accessed. Resource protections remained unchanged.

## Scientific decision

A one-variable attention-logit temperature pilot remains scientifically defensible but is not authorized by this mission. Before any such pilot, owner review should decide whether the nonlocal finite-difference qualification is sufficient or whether a higher-precision/autodiff oracle is required.

If a pilot is approved, it must remain single-variable, bounded, and subject to routing and independent-corpus success criteria. Otherwise conventional retrieval should be retained and the associative ranking investigation closed.

LITE-008 remains blocked.
