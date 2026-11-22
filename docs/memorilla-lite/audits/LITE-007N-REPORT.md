# LITE-007N Read-Only Attribution and Retrieval Diagnostics

## Scope and integrity

LITE-007N was executed from baseline `e8b4e60` on `feature/memorilla-lite-android`. The working tree was clean at preflight. The committed LITE-007L manifests, prepared artifacts, encoder identities, and final checkpoint hashes were verified before analysis.

No optimizer updates were authorized or performed. No replacement checkpoints were created. No holdout data was accessed. All model operations were frozen forward passes or native gradient calculations whose results were discarded.

Reproducible command:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python benchmarks/lite/run_lite007n_diagnostics.py
```

## Evidence integrity

Verified identities are recorded in `LITE-007N-EVIDENCE-INTEGRITY.json`:

- Baseline: `e8b4e60`
- Independent prepared artifact identity: `cf33381044617b285cedf97d08860ad64129d821fd9000625466912ef5fc0252`
- Larger validation artifact identity: `160e2f11ddac255e7006dc53d2b804cc86177f621713bb8c37f157694c308feb`
- Independent dataset SHA-256: `88845e6ac3a01f5216ad0ea246e877114892dff488c84357f0c69180d8b92237`
- Independent manifest SHA-256: `bb9bfe82e7b864f5009b3875e90b5714e4ef9569641a5aa35e9ee059e6f981f1`
- Pinned encoder manifest identity: recorded in the evidence artifact
- Associative and direct final checkpoint hashes: recorded in the evidence artifact

The LITE-007L checkpoint JSON/NPZ hashes continued to match the committed checkpoint manifest.

## Ranking margins

On the independent corpus, the mean best-positive-minus-best-negative margin was:

| Model | Mean margin | Median margin | Mean score variance | Top-1 rate |
|---|---:|---:|---:|---:|
| Associative | -0.02927 | -0.03252 | 0.0002668 | 0.1222 |
| Direct | 0.000267 | 0.000278 | 0.0000000834 | 0.8861 |
| Cosine | -0.01629 | -0.01534 | 0.0001184 | 0.1472 |

The associative model generally places the strongest negative above the strongest positive. Direct scores have very small absolute variance but a positive margin on average and high top-1 accuracy. This confirms that raw score scale is not itself the endpoint; relative ordering is the relevant quantity.

The complete per-query and stratified results are in `LITE-007N-SCORE-MARGINS.json`. Stratifications include collection size, category, required-evidence count, and correct/incorrect top-1 cases. The larger validation corpus is reported separately there.

## Gradient and Jacobian attribution

Native associative gradients were calculated for all 360 eligible independent examples. Results were finite for every example. No update was applied.

Selected mean raw parameter-gradient norms:

| Tensor | Mean norm |
|---|---:|
| `b0.q_w` | `1.79e-05` |
| `b0.k_w` | `3.16e-05` |
| `score_slot_w` | `0.1370` |
| `score_mem_w` | `0.2440` |
| `mem_k_w` | `0.0764` |
| `mem_v_w` | `0.0764` |
| `slots` | `0.0960` |

All connected tensors had nonzero incidence in the analyzed examples. Query/key attention projection gradients were orders of magnitude smaller in raw parameter norm than score and memory projection gradients. A bounded first-20-example sample had aggregate-gradient directional alignment mean `0.2271`, with range `-0.3455` to `0.4778`.

This is evidence for a measurable path-conditioning bottleneck, not proof of optimization failure: raw norms depend on tensor dimensions, parameterization, normalization, and scale. A normalized Jacobian or controlled one-variable intervention would be required to establish causality.

The direct analytical gradient norm mean was `0.1181`; it is not directly comparable to the associative global norm because the parameterizations and objectives differ in path structure.

## Slot representation diagnostics

Across 360 independent examples:

- Mean attention entropy: `2.77258`
- Mean required-evidence enrichment relative to uniform: `0.99876`
- Mean pairwise slot cosine similarity: `0.58628`
- Mean query-conditioning sensitivity: `7.99429`

The enrichment value near one means attention is approximately uniform relative to the required-evidence proportion. Query conditioning exists, but it does not produce strong required-evidence concentration. Slot similarity is substantial, consistent with limited specialization. These observations support weak evidence-selective routing, while leaving architecture versus optimization unresolved.

## Dataset shortcut analysis

The independent corpus contains:

- 90 scenarios
- 360 queries
- 1,680 records
- 6 normalized-template collision groups
- 90 scenarios participating in collision groups
- 270 two-evidence queries and 90 one-evidence queries
- 0 expected-abstention queries

All independent scenarios participate in at least one normalized-template collision group. Therefore collision-group-excluded performance has a zero denominator and is unavailable; it cannot be used as a clean sensitivity estimate. This is a limitation, not proof of leakage.

The larger validation and training artifacts share their source dataset identity, while the independent artifact has separate dataset and embedding identities. The audit reports construction and identity relationships without labeling distribution similarity as proven leakage.

## Diagnostic decision

| Mechanism | Classification |
|---|---|
| Attention-routing gradient attenuation | Supported as a measurable bottleneck; causal effect unresolved |
| Weak evidence-selective slot representation | Supported |
| Direct advantage caused primarily by zero-initialization ties | Contradicted for the final model |
| Larger-validation shortcut or distribution mismatch | Supported as an evaluation limitation; specific leakage unproven |
| Single unique root cause isolated | Unresolved |

## Smallest falsifiable intervention

The smallest next step is still read-only: normalize gradient/Jacobian attribution by parameter scale and measure score-path sensitivity before any training authorization. If the attention-routing attenuation survives scale normalization, authorize at most one single-variable pilot of no more than 40 updates per arm, retaining the existing architecture, corpus, scheduler, and resource guards.

If normalized attribution does not isolate a bottleneck, retain conventional retrieval and do not expand associative training. No multiple-factor optimizer or architecture change is justified by current evidence.

## Resource usage and limitations

The diagnostic process used one numerical thread and one computational process. Peak process usage was recorded in `LITE-007N-EVIDENCE-INTEGRITY.json`; no resource guard was relaxed. The investigation is CPU/NumPy-only and introduced no dependency.

Limitations:

- Gradient norms are raw and require scale-normalized follow-up.
- Direction alignment uses a bounded first-20-example sample.
- Collision exclusion is unavailable because all independent scenarios participate in collisions.
- No final holdout was accessed, so real-world generalization is not assessed.
- No causal architecture or optimizer conclusion can be made from frozen checkpoints alone.

LITE-008 remains blocked pending owner review.
