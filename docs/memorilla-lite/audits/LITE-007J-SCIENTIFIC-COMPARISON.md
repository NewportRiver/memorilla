# LITE-007J Scientific Comparison

## Scope

The experiment consumed exactly 240 associative and 240 direct-bilinear optimizer updates. Formal validation uses 480 validation queries: 240 at 64 records and 240 at 128 records. No holdout was accessed.

Ranking loss is unavailable in the persisted validation ledgers. The reported complete-required-evidence and multi-evidence values are the same top-5 complete-evidence measure, not a separate latent metric.

## Final validation table

All rows use denominator 240 per collection size.

| Model | Collection | R@1 | R@3 | R@5 | MRR | Complete required top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Lexical | 64 | 0.2500 | 0.2500 | 0.2500 | 0.0338 | 0.0000 |
| Lexical | 128 | 0.2500 | 0.2500 | 0.2500 | 0.0181 | 0.0000 |
| Direct cosine | 64 | 0.2500 | 0.2500 | 0.2500 | 0.1488 | 0.2458 |
| Direct cosine | 128 | 0.2500 | 0.2500 | 0.2500 | 0.1214 | 0.2250 |
| Hybrid | 64 | 0.2500 | 0.2500 | 0.2500 | 0.1456 | 0.2458 |
| Hybrid | 128 | 0.2500 | 0.2500 | 0.2500 | 0.1044 | 0.2000 |
| Untrained associative | 64 | 0.3667 | 0.6375 | 0.7000 | 0.2581 | 0.5167 |
| Untrained associative | 128 | 0.2625 | 0.5625 | 0.6167 | 0.1665 | 0.3542 |
| Trained associative | 64 | 0.5500 | 0.7500 | 0.7500 | 0.4014 | 0.7500 |
| Trained associative | 128 | 0.3292 | 0.7500 | 0.7500 | 0.3213 | 0.7500 |
| Untrained direct bilinear | 64 | 0.1708 | 0.2208 | 0.2500 | 0.0637 | 0.0417 |
| Untrained direct bilinear | 128 | 0.1542 | 0.2375 | 0.2583 | 0.0584 | 0.0667 |
| Trained direct bilinear | 64 | 0.7500 | 0.7500 | 0.7500 | 0.2579 | 0.2500 |
| Trained direct bilinear | 128 | 0.7417 | 0.7500 | 0.7500 | 0.2518 | 0.2500 |
| Query-independent ablation | 64 | 0.4167 | 0.5125 | 0.5125 | 0.1683 | 0.1750 |
| Query-independent ablation | 128 | 0.2375 | 0.4750 | 0.4875 | 0.1231 | 0.1625 |

## Paired associative minus direct differences

Pairing is by identical query ID, with 240 pairs per collection size. The intervals are normal-approximation intervals over per-query paired differences, not independent-scenario bootstrap intervals.

| Collection | Metric | Difference | Approx. 95% interval |
|---:|---|---:|---:|
| 64 | R@1 | -0.2000 | [-0.2507, -0.1493] |
| 64 | R@3 | 0.0000 | [0.0000, 0.0000] |
| 64 | R@5 | 0.0000 | [0.0000, 0.0000] |
| 64 | MRR | +0.1435 | [+0.1033, +0.1837] |
| 64 | Complete top-5 | +0.5000 | [+0.4366, +0.5634] |
| 128 | R@1 | -0.4125 | [-0.4760, -0.3490] |
| 128 | R@3 | 0.0000 | [0.0000, 0.0000] |
| 128 | R@5 | 0.0000 | [0.0000, 0.0000] |
| 128 | MRR | +0.0695 | [+0.0219, +0.1171] |
| 128 | Complete top-5 | +0.5000 | [+0.4366, +0.5634] |

These paired results are internally consistent with the table: associative training improves MRR and complete-evidence retrieval versus the trained direct scorer, while direct bilinear has substantially higher R@1. The metrics reward different ranking behavior; there is no single unambiguous winner.

## Interpretation

The scientific result is inconclusive rather than positive. Associative memory improved from its own initialization and achieved higher MRR/complete-evidence retrieval than the trained direct control, but direct bilinear achieved higher top-1 retrieval. The associative result does not establish a general advantage over conventional retrieval or direct ranking.

The evaluation primarily tests learned evidence ranking (A). It provides limited evidence about query-conditioned latent memory compression (B) through the representation probes. It does not test direct latent language-model conditioning (C), because no decoder integration or external inference was performed.
