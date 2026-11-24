# LITE-008M Numerical Acceptance Contract

Predeclared full-reference criteria:

1. Ordinary token-ID and equivalent external-embedding forward logits must agree within a documented Float32 tolerance.
2. External gradients must be finite and nonzero, with decoder parameter gradients absent.
3. At least three of four deterministic directions must have matching finite-difference signs at a representable perturbation.
4. At least one common perturbation size must achieve relative error <= 0.25 for all directions whose analytic absolute derivative exceeds 1e-4, while also reporting absolute error.
5. BF16/FP32 comparison requires gradient cosine >= 0.99, norm ratio in [0.9,1.1], and coordinate sign agreement >= 0.98 in the same component configuration.

The full-model contract was unexecuted because the FP32 reference exceeded the RSS ceiling. The reduced component met the BF16/FP32 comparison criteria but is not a full-decoder pass.
