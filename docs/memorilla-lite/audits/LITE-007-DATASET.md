# LITE-007 Dataset and Holdout Custody Audit

Status: development corpus established; final holdout evaluation blocked.

## Development corpus

- Dataset version: `lite007-dev-v1`
- Seed: `20261120`
- Scenario groups: 300
- Evidence records: 1,500
- Queries: 1,200
- Training scenarios: 240 (80%)
- Validation scenarios: 60 (20%)
- Training queries: 960
- Validation queries: 240
- Records per scenario: 5
- Embeddings: 1,500 records and 1,200 queries, Float32, 384 dimensions
- Encoder contract: `attention-mask-mean-l2-v1`
- Encoder SHA-256: `4278337fd0ff3c68bfb6291042cad8ab363e1d9fbc43dcb499fe91c871902474`

The partition is by scenario ID, not by paraphrased query. Exact record IDs and query IDs are unique. Scenario-level concatenated-content similarity checks found no cross-partition pairs above the 0.92 SequenceMatcher threshold after removing run numbering. The corpus contains multi-hop, multi-evidence, insufficient-evidence, access-policy, irrelevant-similarity, tool-recovery, dependency, resource, conflicting-recommendation, supersession, malicious-content, and environment-change families.

The original LITE-005 development dataset and all historical reports remain unchanged.

## Holdout custody

`benchmarks/lite/fixtures/holdout_manifest.json` exists and hashes to the recorded descriptor hash. It describes a four-scenario, 40-query sealed holdout, but only the descriptor is present in this checkout. No holdout labels or source file are present, and no independent owner-controlled custody endpoint or authorized evaluation procedure is available in this environment.

Therefore:

- Holdout availability: not demonstrated.
- Holdout immutability: descriptor-only, not demonstrated for labels.
- Training access to holdout labels: none.
- Checkpoint-selection access to holdout labels: none.
- Final sealed-holdout evaluation: **BLOCKED**.

No holdout labels were generated, inspected, copied, or used.
