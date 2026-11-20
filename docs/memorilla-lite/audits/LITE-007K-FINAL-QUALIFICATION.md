# LITE-007K Final Corrective Qualification

Status: corrective qualification complete; LITE-007L is not authorized.

## Root cause

LITE-007J rebuilt the eligible list inside each episode and selected the first eight eligible rows. Episode and subprocess boundaries therefore reset selection. The nominal collection-size schedule changed the requested size, not the query identities. The result was eight repeated examples across 240 updates per model.

## Scheduler qualification

Prepared eligible inventory: 540 total: 180 at size 8, 180 at size 16, 180 at size 32. Two simulated model schedules each reserved 240 updates in 30 eight-update episodes. Persistent state was reloaded across subprocesses, including an injected interruption. Each model reserved 240 unique examples, exactly 80 per collection size, with identical order digest `44730c802652fd08b7e8ebb0eb42565d54c4a051da6d4f3868c1126505e90212`. Parameter mutations: 0. Full evidence is in `LITE-007K-SCHEDULER-QUALIFICATION.json`.

The corrected runners reject missing schedulers for LITE-007L/LITE-008 identities. Legacy fallback remains only for explicitly historical/non-corrected compatibility calls.

## Metrics

Corrected metrics distinguish hit rate (any positive at K), evidence recall (fraction of required evidence retrieved at K), standard MRR (first positive reciprocal rank), complete required-evidence retrieval at K, partial multi-evidence coverage, and abstention accuracy. Hand cases cover single-positive, multi-positive, missing-evidence and expected-abstention rows. MRR is at least Hit@1 on identical answerable queries. Original historical artifacts are preserved.

## Initialization

The historical direct initial checkpoint is a zero matrix. Zero scoring is tie-sensitive: a permutation changed Hit@1 from 1.0 to 0.0 on the larger validation artifact. Identity and cosine were read-only controls and produced the same ranking on this normalized embedding artifact. No control was trained. The proposed primary direct control is explicit zero bilinear initialization, with identity and cosine retained as diagnostic controls.

## Corpus

The retained independent corpus has 90 scenarios, 360 queries, 1,680 records, and 30 scenarios per collection size 8/16/32. It has six categories, 360 positive/negative-eligible queries, positive evidence count 1–2, zero exact content duplicates, six normalized-template collision groups, zero embedding pairs above cosine .995, and no holdout material. All 2,040 unique texts were encoded through the existing pinned MiniLM incremental pipeline. The corpus contains no expected-abstention examples; abstention qualification therefore remains incomplete for this corpus and must be represented by a future balanced corpus or the existing labeled evaluation set.

## Corrected frozen evaluations

Independent development corpus, answerable rows:

- Associative final: Hit@1 0.1167, recall@5 0.2181, MRR 0.2358, complete@5 0.1278.
- Direct final: Hit@1 0.8083, recall@5 0.6653, MRR 0.8785, complete@5 0.3417.
- Cosine: Hit@1 0.1472, recall@5 0.3542, MRR 0.3137, complete@5 0.1861.

Existing larger validation, answerable rows, read-only new evaluation:

- Associative final: Hit@1 0.5861, recall@5 1.0000, MRR 0.7759, complete@5 1.0000.
- Direct final: Hit@1 0.9944, recall@5 0.6667, MRR 0.9972, complete@5 0.3333.
- Cosine: Hit@1 0.3333, recall@5 0.3236, MRR 0.3612, complete@5 0.3139.

These results do not establish the proposed associative superiority claim on the independent corpus. Attention/slot/query diagnostics are preserved in the evaluation artifact. No weights were modified.

## Telemetry and next gate

Both runner result schemas now include absolute peak RSS, RSS delta, available RAM, wall time, numerical thread count, computational subprocess count and resource-guard status. Qualification used read-only inference/simulation only. No optimizer updates, architecture changes, dependency installation, resource-limit relaxation, holdout access, APEX integration, brokerage activity or LITE-008 implementation occurred.

The proposed manifest is `LITE-007L-PROPOSED-EXPERIMENT-MANIFEST.json`. Stop here and await explicit owner authorization.
