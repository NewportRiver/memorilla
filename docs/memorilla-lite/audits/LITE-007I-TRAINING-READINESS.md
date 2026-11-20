# LITE-007I Training Readiness

Status: prepared for owner review; sustained training is not executed by LITE-007I.

## Proposed J configuration

- Global optimizer allowance: 240 updates
- Episode size: at most 8 updates per subprocess
- Sequential episodes: 30
- Concurrent training processes: 1
- Numerical threads: 1
- Evidence collection: 8/16/32 for training; 64/128 evaluation-only
- Optimizer: Float32 SGD, learning rate 0.001, gradient clipping norm 1.0
- Batch size: 1
- Validation: before training, after every 8-update episode, and at predeclared cumulative intervals 48/96/144/192/240
- Checkpoint identity: dataset content hash, encoder identity, architecture version, parameter-subset identity, episode ID
- Sample ordering: deterministic eligible-index order with persisted position
- Resource limits: minimum RAM 1,280 MiB; abort RAM 1,024 MiB; maximum RSS 256 MiB; target below 128 MiB; wall time 30 seconds; storage minimum 5 GiB

## Proposed parameter schedule

1. Episodes 1-5: learned slots and query conditioning, maximum 40 updates total.
2. Episodes 6-10: memory-summary projections, maximum 40 updates total.
3. Episodes 11-20: attention Q/K/V/output and layer normalization, maximum 80 updates total.
4. Episodes 21-30: feed-forward refinement parameters, maximum 80 updates total.

Each subset must remain frozen outside its episode authorization. `out_w` and `out_b` remain disconnected from the ranking objective and are excluded.

## Stopping criteria

Stop immediately on:

- Any resource guard violation
- Nonfinite loss or gradient
- Frozen-parameter checksum change
- Checkpoint/budget identity mismatch
- No checkpoint after an authorized update
- Validation ranking degradation beyond a predeclared tolerance for two consecutive checks
- Abstention or policy-violation regression
- Any missing telemetry phase

No automatic retry follows resource exhaustion or checkpoint corruption.

## Controls

The associative model must be compared with lexical, direct cosine, hybrid, untrained neural, scoring-head-trained, memory-projection-trained, joint, direct bilinear, query-independent, and query-conditioned controls. The direct bilinear control must be separately authorized and labeled independently from associative training.

## Current qualification boundary

The 8/16/32/64/128 corpus and pinned embeddings are present. The complete LITE regression passes. One attention/FFN feasibility episode passed with full telemetry. The expanded neural comparison remains incomplete because the bounded Android evaluation exceeded 180 seconds; only lexical, direct cosine, and hybrid results are currently certified for all five collection levels.

Owner approval is required before any J optimizer update.
