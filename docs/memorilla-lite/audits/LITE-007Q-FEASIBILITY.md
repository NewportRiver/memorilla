# LITE-007Q Gate A Feasibility Report

## Scope and authorization

LITE-007Q Gate A was executed from baseline `dd8bec3` on `feature/memorilla-lite-android`.

This was a frozen-checkpoint diagnostic sweep only:

- Optimizer updates: `0`
- Gate B training: not authorized and not performed
- Checkpoints: not replaced or modified
- Holdout: not accessed
- Encoder, embeddings, objective, architecture, and parameter count: preserved
- Threads: `1`
- Computational subprocesses: `1`
- Peak RSS: `86,132 KiB` in the feasibility process

The original LITE-007L result remains unchanged and is not reinterpreted by this sweep.

## Implementation

`NeuralConfig` now exposes `attention_logit_scale`, defaulting to `1.0`. The scale is applied to the attention dot-product logits in:

- Normal forward attention
- Streaming forward attention
- Native reverse-mode gradients
- Memory-projection gradient path

Scale `1.0` preserves the original operation. The parameter count remains `429,312`; the scale is configuration, not a trainable tensor.

Reproducible command:

```sh
PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python benchmarks/lite/run_lite007q_feasibility.py
```

## Scale-1 compatibility

The frozen final associative checkpoint was evaluated with its original configuration and with an explicit `attention_logit_scale=1.0` configuration using identical state arrays and queries.

- Maximum forward output/latent difference: `0.0`
- Parameter count unchanged
- Scale-1 regression test: passed

## Frozen diagnostic sweep

Predeclared scales: `0.5`, `1.0`, `2.0`, `4.0`.

| Scale | Hit@1 | Mean margin | Required-evidence enrichment | Mean entropy | Mean logit spread |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.1222 | -0.0292746 | 0.99938 | 2.772587 | 0.02079 |
| 1.0 | 0.1222 | -0.0292746 | 0.99876 | 2.772582 | 0.04158 |
| 2.0 | 0.1222 | -0.0292747 | 0.99753 | 2.772562 | 0.08316 |
| 4.0 | 0.1222 | -0.0292747 | 0.99507 | 2.772480 | 0.16633 |

The scale changes logit spread and slightly lowers entropy, as mathematically expected. It does not improve independent Hit@1, mean margin, or required-evidence enrichment in the frozen checkpoint.

This sweep is diagnostic evidence only. No scale was selected as an unbiased final result, and no repeated tuning against the independent corpus is authorized.

## Derivative validation

The scale factor is present in both forward and reverse attention operations. Existing LITE-007P numerical qualification remains applicable to the native reverse-mode implementation. Additional frozen checks were run at each scale using the same FP32 centered finite-difference style.

The upstream `b0.q_w` checks remain numerically unstable at the low gradients and Float32 precision involved. This is expected from LITE-007P: the finite-difference loss delta can quantize to zero or one ULP. The scale-1 forward identity is exact, the scale-dependent logit spread is monotonic, and scale-1 forward/gradient contract tests pass. No derivative defect was identified.

The complete raw feasibility and derivative records are in `LITE-007Q-FEASIBILITY.json`.

## Gate A decision

Gate A is technically feasible:

- Configurable scaling is implemented without architecture expansion.
- Scale `1.0` is exactly backward-compatible for frozen outputs.
- The operation changes attention spread under bounded CPU/RSS conditions.
- Regression coverage exists.

Gate A does not establish retrieval benefit. The frozen sweep produced identical Hit@1 at all scales, with enrichment remaining near uniform. Therefore no Gate B training pilot is recommended automatically.

## Gate B status

Gate B remains NOT AUTHORIZED. No training, optimizer updates, new budgets, matched exposure, or pilot checkpoints were created.

If the owner separately authorizes Gate B, it must use one predeclared intervention scale, a maximum of 40 updates per arm, matched examples and seeds, and predeclared mechanistic and independent-corpus success criteria. Otherwise, retain conventional retrieval and leave LITE-008 blocked.
