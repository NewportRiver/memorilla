# LITE-008G Scientific Limitations

1. The development set has only eight entities; confidence intervals are bootstrap descriptive intervals, not population guarantees.
2. The decoder performed substantially better with relevant ordinary textual evidence, establishing task capability but not proving that the selected prompt format is optimal for latent inputs.
3. All greedy exact-answer accuracies were zero, including relevant textual evidence, so NLL is the more informative signal in this small one-token setup.
4. Historical and corrected projections were intentionally trajectory-equivalent under scaled learning rates; this does not establish that the original optimization was semantically valid.
5. The corrected projection improved NLL over query-only, but constant mean, mismatched, and zero controls also improved NLL. The effect is therefore not attributable to factual projected memory.
6. Counterfactual projected memory favored the supplied value in only 8/16 directions; the bootstrap interval included zero.
7. The oracle condition uses genuine decoder answer-token embeddings and is diagnostic only.
8. Aggregate semantic decoder RSS was not directly sampled. Independent projected and zero harness probes measured 185,942,016 and 186,310,656 bytes respectively.
9. The instrumentation retry hung at the first case and was killed; no result from that retry is included.
10. No protected final holdout, production retrieval, SQLite evidence, MemoryModule integration, decoder fine-tuning, GPU, cloud inference, or unrestricted optimization was used.
11. Thermal and separate KV-cache telemetry were unavailable.
