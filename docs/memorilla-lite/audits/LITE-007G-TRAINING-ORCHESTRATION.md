# LITE-007G Training Orchestration

`lite.training.orchestration.EpisodeBudget` provides the persistent authorization boundary for future episodes.

Properties:

- Fresh native Python subprocesses may reopen the same budget file.
- Dataset and checkpoint identities are bound at initialization.
- Global update usage persists atomically across subprocess boundaries.
- An episode cannot reserve more updates than remain globally authorized.
- Every update increments persistent usage before the process can claim completion.
- Episode status and completed update count are persisted.
- Temporary files are replaced atomically.
- Identity mismatches fail closed.
- Resource exhaustion must terminate the episode without automatic retry.

LITE-007G did not execute an optimizer episode. The controller was verified only with temporary budget fixtures; no generated weights or production checkpoint were created.

Future authorized episode configuration must retain:

- `OMP_NUM_THREADS=1`, one computational subprocess
- 1,280 MiB minimum available RAM
- 1,024 MiB abort RAM
- 256 MiB maximum RSS, target below 128 MiB
- 30 second maximum wall time
- 32 maximum evidence records per training example
- Float32 SGD, no persistent moments unless separately approved
- Atomic checkpoint and metadata writes
- Dataset/checkpoint identity verification before reservation
