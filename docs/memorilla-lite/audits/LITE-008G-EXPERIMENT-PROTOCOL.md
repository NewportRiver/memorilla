# LITE-008G Experiment Protocol

## Predeclared acceptance criteria

- Textual-evidence prerequisite: relevant evidence must improve mean target NLL over query-only on calibration examples; otherwise stop for decoder-format investigation.
- Projection success: corrected projection must improve over query-only, untrained projection, and constant-mean embedding with non-overlapping practical evidence on entity-disjoint development examples.
- Counterfactual success: supplied-value likelihood must favor the supplied value over the alternate in a strong majority of both directions, with a positive paired margin and uncertainty interval excluding zero.
- Exact-answer success: greedy exact-answer accuracy must be non-zero and improve over query-only.
- Unsupported behavior: unanswerable/query-only controls must not be represented as factual success.
- Resource: decoder RSS remains below 268,435,456 bytes, CPU-only, one numerical thread, one native subprocess at a time; no whole-pipeline claim without direct phase evidence.
- Holdout: protected final holdout remains inaccessible.

Criteria were fixed before inspecting development results.

## Dataset

Seed `20261124`; 32 randomized entities and randomized assignments over eight color values. Partitions are entity-disjoint: 16 train, 8 calibration, 8 development. Two question templates alternate by row; values are not sequentially mapped to entity names. The final protected holdout was not created or accessed.

Projection training uses 16 train facts. Historical mode uses the preserved unscaled gradient at `lr=0.08`; corrected mode uses the true MSE gradient at `lr=23.04`, which reproduces the historical update scale while making the objective mathematically correct.

## Conditions

Calibration: query-only, relevant textual evidence, irrelevant textual evidence, counterfactual textual evidence.

Development: query-only, zero, fixed-seed random, untrained projection, constant mean target embedding, historical projection, corrected projection, mismatched projection, counterfactual projected memory, and genuine answer-token embedding oracle. The oracle is diagnostic only and not deployable.

The semantic raw run performed 112 teacher-forced cases and 112 separate one-token greedy-generation cases. A separate 32-case teacher-forced counterfactual matrix scored both candidate values under both supplied projected memories.
