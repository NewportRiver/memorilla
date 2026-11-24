# LITE-008M Dataset Correction Audit

LITE-008L used deterministic indexing that produced only two effective randomized answer combinations, with the original correct answers concentrated in blue-family values. That made query-only preferences and evidence conditioning difficult to interpret.

LITE-008M preserves all LITE-008L artifacts and creates a new manifest. The corrected dataset uses seeded shuffling, independent randomized entity identities, balanced single-token value frequencies, balanced A/B orientations, counterbalanced candidate ordering, entity-disjoint calibration/development partitions, and four separately marked multi-token pairs.

The primary development denominator contains 24 single-token pairs. The multi-token cases are reported separately rather than mixed into the primary result.
