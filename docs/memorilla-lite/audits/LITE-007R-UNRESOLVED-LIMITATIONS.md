# LITE-007R Unresolved Limitations

## Evaluation and comparison

- LITE-007L's decisive independent comparison is a development corpus, not a final holdout. No final holdout was accessed.
- Conventional LITE-006/LITE-007J artifacts use different denominators and collection-size protocols from LITE-007L. Their values are preserved but not treated as interchangeable.
- Dataset audit found normalized-template collision groups involving all 90 independent scenarios. This is an evaluation limitation and possible shortcut risk, not proof of leakage.
- Multi-evidence completeness is not equivalent to correct selective ranking. Some reported complete-support values are conditional on required-evidence subsets.
- Direct bilinear, cosine, and hybrid latency/RSS are not all measured under one identical harness. Missing values remain unavailable.

## Associative model

- The associative model's negative independent result establishes failure of this tested ranker/training setup, not impossibility of latent memory conditioning.
- Near-uniform attention, weak enrichment, slot similarity, and negative score margins support a routing/conditioning problem but do not identify a unique causal defect.
- LITE-007Q altered attention sharpness without improving frozen retrieval. It did not test a trained intervention, and Gate B was not authorized.

## Numerical gradients

- LITE-007P qualified all 27 ranking-connected tensors using a bounded FP32 fixture and a relatively large centered perturbation. Small upstream finite differences remain susceptible to cancellation and ULP quantization.
- LITE-007Q's upstream `b0.q_w` finite differences remained unstable across scales: some numerical deltas were zero and one scale produced a large relative discrepancy. This is an explicit limitation, not evidence of a new implementation defect.
- Scale-1 forward compatibility was exact and reverse-mode paths were covered by regression, but higher-precision or independent autodiff confirmation would strengthen future causal attribution.

## Product and systems

- No language-model decoder conditioning was implemented or evaluated.
- No cloud/GPU scaling, external inference, APEX integration, or production persistence migration was evaluated.
- Android compatibility is CPU/resource feasibility evidence, not full device acceptance.
- No claim is made that the current conventional baseline is globally optimal; it is the best-supported operational choice under this record.
