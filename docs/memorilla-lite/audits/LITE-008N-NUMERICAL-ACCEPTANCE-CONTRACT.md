# LITE-008N Numerical Acceptance Contract

Retained from LITE-008M unless explicitly marked below:

1. Reduced conventional/reference loss absolute error <= 1e-6.
2. Reduced final-logit maximum absolute error <= 1e-5.
3. Reduced external-gradient cosine >= 0.99.
4. Reduced gradient norm ratio in [0.9, 1.1].
5. Complete FP32 finite differences use four deterministic directions and require at least one common perturbation size with relative error <= 0.25 in every direction with analytic derivative magnitude above 1e-4.
6. Full BF16 comparison requires cosine >= 0.99, coordinate sign agreement >= 0.98, norm ratio in [0.9,1.1], and reports relative L2 error and loss difference.

Results: reduced equivalence passed exactly; complete FP32 finite differences passed at epsilon 0.25; full BF16 cosine and norm ratio passed, but coordinate sign agreement `0.9704861` failed the retained `0.98` threshold. No threshold was changed retrospectively.
