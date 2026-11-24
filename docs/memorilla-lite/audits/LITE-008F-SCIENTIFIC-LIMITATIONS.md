# LITE-008F Scientific Limitations

1. The optimized objective reconstructs decoder input embeddings. It does not optimize target-answer likelihood, teacher-forced sequence likelihood, or counterfactual answer margin.
2. The synthetic development set is small and template-driven. It demonstrates entity-disjoint reconstruction behavior, not generalization to natural historical memory.
3. The downstream decoder probe used one projected representation and a short continuation. Different token sequences are sensitivity evidence, not semantic correctness.
4. The probe did not establish that the generated text answers the corresponding synthetic fact, beats query-only or ordinary-text evidence baselines, or resists distractors.
5. MiniLM feature extraction peaked above the approved 256 MiB ceiling in the combined process. It was measured separately from bridge training and decoder inference; no whole-pipeline compliance claim is made.
6. The pinned Q4 runtime exposes inference but not decoder backpropagation. Methods A and C were not executed.
7. Thermal telemetry and separate KV-cache allocation were unavailable.
8. Greedy deterministic runs do not establish stochastic seeded reproducibility.
9. Existing `out_w/out_b` memory output parameters remain untrained and were not repurposed. Existing LITE-007 checkpoints remain untouched.
10. No protected holdout, SQLite evidence, production retrieval path, complete associative-memory training, learned 384-to-576 memory integration, or Gate D/E/F qualification was performed.
