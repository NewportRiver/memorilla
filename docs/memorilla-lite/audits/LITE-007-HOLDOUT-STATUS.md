# LITE-007B Holdout Status

Status: blocked; only a descriptor is present in this checkout.

## Verified state

`benchmarks/lite/fixtures/holdout_manifest.json` exists as a descriptor and is referenced by the prior dataset audit. The descriptor describes a four-scenario, 40-query sealed holdout. No holdout payload or labels are present in the repository checkout. No owner-controlled custody endpoint or authorized release procedure was discoverable during this recovery.

No holdout labels were read, generated, copied, used for training, used for checkpoint selection, or used for evaluation.

## Integrity conclusion

The descriptor hash does not establish that the sealed payload exists or that its labels remain immutable. Holdout availability and payload immutability are therefore not demonstrated. The final sealed-holdout evaluation remains blocked.

## Required independent creation procedure

If the owner authorizes creation of a real holdout:

1. Generate scenarios with a fresh recorded seed and a source manifest distinct from the development corpus.
2. Verify scenario IDs, record IDs, query IDs, access scopes, timestamps, and source hashes are disjoint from all development and training artifacts.
3. Write the payload once to owner-controlled storage, calculate a cryptographic hash, and retain the payload outside the repository and training runtime.
4. Commit only a descriptor containing schema, counts, creation timestamp, hash, custody identifier, and release policy; never commit labels or payload.
5. Make training and checkpoint-selection code fail closed when the holdout path or labels are visible.
6. Release the payload only after model freeze and checkpoint selection, through an owner-mediated evaluation procedure that records the exact checkpoint hash and holdout hash.
7. Preserve an append-only evaluation report and prohibit tuning or retraining after label release.

Until these steps are completed and independently verified, the holdout gate is blocked and LITE-007 cannot be complete.
