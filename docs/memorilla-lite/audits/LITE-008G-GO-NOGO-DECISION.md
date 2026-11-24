# LITE-008G GO / NO-GO Decision

## Decision: NO-GO for the evaluated projection as meaningful semantic memory

The mathematical and evaluation infrastructure is now established:

- Correct MSE gradients verified for weights and bias across multiple batch sizes and output widths.
- Historical update behavior preserved under an explicit legacy method.
- Corrected learning-rate scaling reproduced the historical trajectory.
- Teacher-forced target scoring implemented and validated against ordinary token/embedding equivalence.
- Encoder resource diagnosis completed without replacing MiniLM or adding dependencies.
- Entity-disjoint randomized dataset and fixed partitions recorded.
- Textual evidence prerequisite passed: relevant evidence NLL `5.1122` versus query-only `9.0973` on calibration.

The projection itself fails the predeclared semantic criteria:

- Corrected projection development NLL: `7.9258`.
- Query-only: `8.7520`.
- Untrained projection: `8.6990`.
- Constant mean embedding: `8.1133`.
- Mismatched projection: `8.0886`.
- Oracle answer embedding: `7.5433`.
- Corrected projection greedy exact accuracy: `0/8`.
- Counterfactual supplied-value margin: positive in `8/16`, mean `+0.0458` nats, 95% bootstrap interval `[-0.5042, +0.5920]`.

Although corrected projection NLL improved over query-only and untrained controls, constant, mismatched, and zero controls also improved. The result does not demonstrate factual information transfer through the learned projection. Counterfactual sensitivity is inconsistent.

This is a projection-method NO-GO, not a decoder-capability failure: relevant textual evidence materially improved likelihood. It is also not a whole-pipeline resource failure: MiniLM and bridge stages were within the isolated ceiling, while aggregate semantic decoder RSS was not directly measured. The valid independent decoder qualification probe remained below the ceiling.

No production memory architecture, checkpoints, decoder weights, retrieval, SQLite evidence, or holdout data changed. Gate D, Gate E, and Gate F remain unpassed. Stop at owner review before any decoder-likelihood optimization or joint associative-memory training.
