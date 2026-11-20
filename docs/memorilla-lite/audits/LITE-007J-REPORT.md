# LITE-007J Authorized Comparative Training Report

Status: authorized experiment executed; LITE-007 remains scientifically limited and is not a production or holdout certification.

## Authorization and execution

The owner directive authorized two non-transferable 240-update budgets. Both were consumed exactly:

- Associative memory: 240 updates, 30 completed eight-update episodes, 0 interrupted.
- Direct bilinear ranking: 240 updates, 30 completed eight-update episodes, 0 interrupted.
- Combined: 480 updates.

Previous H/I feasibility updates were not transferred.

## Recovery qualification

A real disposable subprocess was terminated at the boundary after `.npz` publication and before metadata publication. The mismatched pair was rejected by normal loading. Explicit recovery restored the previous checkpoint generation and reloaded it successfully. No automatic retry occurred. The last valid generation remained recoverable.

## Associative qualification

The fresh untrained initialization completed the required eight-update qualification with all 27 ranking-connected tensors selected. `out_w` and `out_b` remained byte-identical. Initial/final qualification losses were `0.6521326899528503` and `0.6343272924423218`. Native reload passed and all finite/resource checks passed.

## Training curves

The associative per-episode loss declined across the run, but the final update within an episode was not monotonically lower than the first update in every episode. The direct bilinear loss declined gradually from approximately `0.744397` to `0.738132` across its episode sequence.

Full milestone curves are in `LITE-007J-LEARNING-CURVES.json` for 0, 40, 80, 120, 160, 200 and 240 updates, using validation-only denominators of 480 queries per model.

## Validation comparison

Validation queries were restricted to the independent development-validation partition:

- 64-record collections: 240 queries
- 128-record collections: 240 queries
- Total validation denominator per model/milestone: 480

The resumable controller completed every model/milestone evaluation without partial aggregation. Conventional lexical, cosine and hybrid results were recomputed on those same validation queries.

The trained associative model did not establish a general superiority claim. At 240 updates, direct bilinear retrieval had high top-1 retrieval on this generated validation set, while associative and conventional metrics varied by collection size and metric. These results are development evidence only and are not holdout evidence.

## Representation analysis

Weight-level snapshots and finite checks are recorded. Slot-specific diversity and attention-distribution analysis were not exposed by the current native checkpoint/forward schema, so no unsupported latent interpretation is claimed. No cognitive-improvement claim is made.

## Resources

Associative telemetry:

- Peak RSS: `124,809,216` bytes
- Minimum available RAM: `1,425,125,376` bytes
- Maximum episode wall time: `1.1907247659983113s`
- Numerical threads: `1`
- Computational subprocesses: `1`
- Target RSS below 128 MiB: passed
- RSS ceiling 256 MiB: passed

Direct runner resource telemetry was not instrumented at episode phase granularity. This is a reporting limitation; no resource guard violation was observed, but direct telemetry is not retrospectively fabricated.

## Checkpoints

Generated weights remain outside Git. Final NPZ hashes and identity metadata are in `LITE-007J-CHECKPOINT-MANIFEST.json`. Frozen output projection hashes are identical before and after associative training.

No holdout access occurred.

## Regression

Updated complete LITE suite:

- 76 passed
- 0 failed
- 0 skipped
- 0 timed out
- 0 unexecuted
- 7.19 seconds

## Scientific limitations

- The dataset is synthetic/generated and the normalized-template audit is bounded; it does not establish semantic independence.
- No final holdout evaluation was performed.
- Direct-model episode resource telemetry is incomplete.
- Representation analysis is limited to checkpoint weight norms and finite checks.
- The direct bilinear implementation used a deterministic first-positive/first-negative pair per eligible example; this is documented and was matched across its own exposure, but is not equivalent to all possible negative sampling policies.
- No production, Android-device, external-inference or APEX evaluation occurred.

## Disposition

LITE-007J controlled training executed successfully under the authorized budget. The result does not justify a claim that query-conditioned associative memory improves retrieval over direct bilinear or conventional controls. The next step is owner review of the negative/ambiguous comparative result and the direct telemetry limitation before any LITE-008 design.
