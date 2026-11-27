# LITE-008W-R4 Control Implementation Audit

R3's `minimum_norm` label was incorrectly assigned the selected ridge matrix. R4 recomputed both matrices independently. Their hashes and numerical differences are recorded in `LITE-008W-R4-PROJECTION-COMPARISON.json`; they differ materially, although both retain 8/8 factual reversals.

R3's `query_only`, `zero_memory`, and `untrained_projection` controls were numerically EOS-equivalent because query-only and zero-residual untrained projection both passed the EOS slot to the decoder. R4 separates them: `zero_embedding` is an actual all-zero external vector; `query_only` is EOS-slot; `untrained_residual_projection` is explicitly documented as EOS-equivalent.

R4 also distinguishes same-entity opposite-fact evidence from unrelated-entity evidence, uses a seeded norm-matched random vector, and computes mean teacher only from training artifacts.
