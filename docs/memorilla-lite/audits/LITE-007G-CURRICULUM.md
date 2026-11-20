# LITE-007G Curriculum Specification

This is a development-only curriculum specification. It does not create or access sealed holdout data and does not authorize optimizer execution.

## Collection levels

| Level | Collection size | Use |
|---|---:|---|
| 1 | 8 | train and validation |
| 2 | 16 | train and validation |
| 3 | 32 | train and validation; maximum current training budget |
| 4 | 64 | evaluation only |
| 5 | 128 | evaluation only |

The existing LITE-007 corpus has five records per scenario and therefore cannot challenge larger collections. The current prepared artifact is retained unchanged. A future curriculum generator must use existing verified encoder/preprocessing identities and write a separate content-addressed artifact.

## Required scenario families

- direct factual retrieval
- relationships across separate records
- multiple independent supporting facts
- conflicts, corrections, and supersession
- multi-step association
- lexical and semantic hard negatives
- insufficient evidence and abstention
- misleading or malicious memory content as inert data
- temporal dependencies
- cross-document associations
- prior failed approaches and applicable lessons

## Integrity requirements

- Scenario-disjoint train/validation partitioning
- No answer text used as an input feature
- No near-duplicate scenario or query across partitions
- No required evidence ID crosses partition boundaries
- No future record survives historical cutoff filtering
- No holdout payload or labels in the training workspace
- Source, encoder, preprocessing, and curriculum hashes persisted
- Eligibility and evidence identities resolved before model scoring

## Scientific limitations to correct

LITE-007F's Recall@5 was saturated at 1.0 because every current scenario has at most five records. The eight-update pilots also expose only eight examples and should not be treated as meaningful curriculum coverage. Abstention performance was not reported in the earlier ranking-loss pilot; future reports must include abstention accuracy, false abstention, unsupported retrieval, and policy violations with explicit denominators.
