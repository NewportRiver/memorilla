# LITE-008S numerical metric contract

Metrics: absolute max/L2 error, reference-relative error, symmetric-relative error, cosine and gradient norm ratio. No denominator floor of one. Non-negligible derivatives use reference-relative threshold `0.10`; near-zero derivatives use absolute tolerance `1e-3`. Stage A uses Float64 weight/bias closed forms, autograd and central differences. Stage D records epsilons `.001`, `.01`, `.1`; `.1` is the qualification epsilon because BF16 forward quantization dominates smaller perturbations.
