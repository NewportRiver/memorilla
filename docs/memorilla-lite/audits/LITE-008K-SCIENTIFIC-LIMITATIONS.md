# LITE-008K Scientific Limitations

- The multi-pair counterfactual oracle failed its preregistered gate: only `1/4` pairs reversed candidate preference.
- The first oracle implementation timed out after 300 seconds without a durable phase ledger. Its exact update count is unavailable and is not presented as measured success.
- The corrected rerun used Float32 master slots, Float32 loss, complete tokenization, and dedicated positions, but BF16-cast finite differences remained inconsistent across directions and perturbations; the best full-objective relative error was `0.1889`.
- No generalized semantic bridge was trained. Therefore no development improvement, MSE comparison, soft-prompt comparison, mismatched-memory advantage, or quantized bridge transfer can be claimed.
- Textual calibration was small and calibration-only: relevant evidence ranked correctly on `2/4`, query-only on `1/4`, unrelated on `2/4`, and counterfactual on `0/4`.
- The LITE-008J per-example optimized embedding and Q4 transfer remain historical mechanism evidence, not generalized bridge evidence.
- No protected final holdout was accessed. No production MemoryModule, SQLite evidence, retrieval system, MiniLM artifact, decoder weights, or checkpoint was modified.
- Gate D, Gate E, and Gate F remain unpassed.
