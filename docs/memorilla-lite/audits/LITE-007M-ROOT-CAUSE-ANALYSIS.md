# LITE-007M Root-Cause Analysis

## Scope and baseline

This is a read-only investigation at baseline `ef75802` on `feature/memorilla-lite-android`. The working tree was clean before analysis. The LITE-007L checkpoints and committed audit artifacts were present. No optimizer updates, replacement checkpoints, dependency changes, holdout access, or LITE-008 work occurred.

Primary evidence:

- `LITE-007L-EXPERIMENT-MANIFEST.json`
- `LITE-007L-EXPOSURE-VERIFICATION.json`
- `LITE-007L-LEARNING-CURVES.json`
- `LITE-007L-COMPARATIVE-EVALUATION.json`
- `LITE-007L-REPRESENTATION-ANALYSIS.json`
- `LITE-007L-ORDERING-CONTROLS.json`
- `LITE-007L-ORDERING-SENSITIVITY.json`
- `LITE-007L-RESOURCE-TELEMETRY.json`
- `LITE-007L-INITIALIZATION-AUDIT.json`
- `LITE-007L-AUTHORIZATION-LEDGER.json`
- `LITE-007L-CHECKPOINT-MANIFEST.json`

## Characterization of the gap

On the independent 360-query development corpus at 240 updates:

| Model | Hit@1 | Recall@1 | Recall@5 | MRR | Complete@5 |
|---|---:|---:|---:|---:|---:|
| Associative | 0.1222 | 0.0722 | 0.2250 | 0.2446 | 0.1333 |
| Direct | 0.8861 | 0.5542 | 0.6681 | 0.9351 | 0.3361 |
| Cosine | 0.1472 | 0.0778 | 0.3542 | 0.3137 | 0.1861 |

The direct model is substantially better at first-result ranking and MRR. It is also better on Complete@5, although the gap is smaller than the Hit@1 gap. The associative model is only modestly above cosine on Hit@1 and MRR and below cosine on Recall@5 and Complete@5.

The associative curve is not flat: independent Hit@1 increases from 0.0222 at initialization to 0.1222 at 240 updates, and MRR increases from 0.1599 to 0.2446. This is evidence of learning, but not of sufficient learning or appropriate generalization.

## A. Representation effectiveness

The representation audit shows nonzero query sensitivity, so the model is not completely query-independent. However, required-evidence attention is generally close to the uniform-attention baseline, and slot cosine similarities remain substantial. This is consistent with limited slot specialization and weak evidence-selective routing.

The larger corpus produces a very different result: associative Complete@5 reaches 1.0 at 240 updates, while the independent corpus remains at 0.1333. That divergence is compatible with a representation that can exploit a structured development distribution without preserving robust evidence distinctions across the independent corpus.

This supports a representation bottleneck hypothesis, but does not prove that the architecture is solely responsible. Poor optimization can produce the same observed representation.

## B. Optimization effectiveness

The direct model has a direct query-by-evidence bilinear path. The associative model must propagate the ranking signal through query projection, memory projections, attention, latent slots, and a score head. The learning curves show that this path improves slowly relative to direct ranking under the same nominal update budget.

Telemetry and checkpoint records show finite gradients, finite parameters, valid mutations, and passed resource guards. Therefore the failure is not explained by numerical corruption, resource breach, or absent updates.

The evidence supports optimization/path conditioning as a major contributor, but the committed artifacts do not contain enough gradient-alignment or per-layer attribution data to isolate the exact failing path.

## C. Direct-ranking advantage

Direct ranking is simpler and less bottlenecked. Its final independent Hit@1 is 0.8861 versus 0.1222 for associative ranking. The direct final model is stable under the evaluated evidence-order reversal. The zero direct initialization is order-sensitive only because every initial score is tied: ordinary Hit@1 is 1.0, reversed Hit@1 is 0.0, with 360 tied rows.

Thus the final direct advantage is not adequately explained by the zero-initialization tie artifact. It is most plausibly an optimization and path-complexity advantage, although this is not proof that the direct model is a better general memory architecture.

## D. Dataset and evaluation integrity

The independent corpus contains no expected-abstention examples, so abstention behavior is unavailable there. Six normalized-template collision groups were identified. Final associative metrics were unchanged under the evaluated evidence-order reversal, and final direct aggregate metrics were unchanged apart from six tie rows.

The larger validation corpus reaches a complete-evidence ceiling of 1.0 for the associative model at 240 updates. Because this ceiling is absent on the independent corpus, the larger corpus should not be treated as evidence of generalization without a further shortcut/identity audit. The current evidence establishes a serious evaluation limitation, not proven leakage.

## Root-cause conclusion

No single root cause is demonstrated. The ranked explanation is:

1. Optimization/path conditioning makes the associative route much harder to train with the current objective and budget.
2. The resulting associative representation shows weak evidence-selective routing and limited slot specialization.
3. The direct model benefits from a simpler direct query-evidence interaction.
4. The larger validation corpus likely measures an easier or shortcut-prone distribution relative to the independent corpus.
5. Ordering and collision effects are real evaluation limitations but are unlikely to explain the entire final associative/direct gap.

The highest-confidence conclusion is therefore a multi-factor diagnosis, not an architecture verdict.

## Scientific boundary

LITE-007M does not authorize training, architecture changes, decoder conditioning, or LITE-008. A simpler retrieval architecture with a better accuracy-to-resource tradeoff remains scientifically preferable if controlled evidence supports it.
