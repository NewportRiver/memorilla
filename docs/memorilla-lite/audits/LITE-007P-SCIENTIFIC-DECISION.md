# LITE-007P Scientific Decision

## Qualification status

The associative native reverse-mode differentiation implementation is qualified for the deterministic FP32 fixture at the declared nonlocal perturbation condition:

- 27 / 27 ranking-connected tensors checked
- Centered finite differences
- Loss scale `1e8` for numerical observability
- Perturbation `0.5 × parameter RMS`
- Maximum relative error `< 0.25`
- Zero optimizer updates

Small-perturbation finite differences remain numerically unstable in FP32 and are not accepted as a defect signal without higher-precision arithmetic or a better-conditioned observable.

## Decision on attention attenuation

The LITE-007O conclusion remains valid:

- Attention query/key gradients remain much smaller after parameter-scale normalization.
- Attention entropy remains near-uniform over eligible evidence positions.
- Initial and final routing statistics remain effectively unchanged.
- The discrepancy did not reveal a verified reverse-mode implementation defect.

Therefore, the attention-path attenuation hypothesis remains supported as a measurable bottleneck, with causal confidence limited by parameterization and the finite-difference conditioning boundary.

## Decision on a temperature pilot

A one-variable attention-logit temperature pilot remains scientifically justifiable as a future, separately authorized diagnostic intervention. It is not authorized by LITE-007P.

The pilot must not be expanded into broad associative retraining. It should be rejected if it fails to improve both routing observables—attention-logit spread and required-evidence enrichment—and independent-corpus ranking margins under unchanged resource and exposure limits.

If owner review does not authorize that narrowly bounded pilot, retain conventional retrieval and close the associative ranking investigation.

LITE-008 remains blocked.
