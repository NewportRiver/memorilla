# LITE-007J Generalization Audit

## Partition and collection sizes

The dataset declares 180 training scenarios and 120 validation scenarios. Training eligibility is limited to collection sizes 8, 16 and 32. Validation for the milestone curves is limited to 64 and 128 records, with 240 queries at each size.

This cleanly separates larger-collection validation from the training partition, but it does not isolate scenario generalization from collection-size scaling: both scenario identities and collection size change between training and formal validation.

## Actual training exposure

The persisted run ledgers show 240 updates per model but only 8 unique query identities repeated throughout training:

- 3 examples at size 8: 90 updates total
- 3 examples at size 16: 90 updates total
- 2 examples at size 32: 60 updates total

The associative and direct models used identical order, eligible collections, required-evidence labels, objective configuration and update count. However, this is repeated-example exposure, not broad curriculum coverage. The manifest's intended deterministic ordering was preserved, but the actual schedule did not expose the full eligible training population.

## Read-only training-partition inference

Final checkpoints were evaluated without optimizer updates on 720 training-partition queries, 240 per collection size:

| Model | Size | R@1 | R@3 | R@5 | MRR | Complete top-5 |
|---|---:|---:|---:|---:|---:|---:|
| Associative final | 8 | 0.4625 | 0.7500 | 0.7500 | 0.3833 | 0.7500 |
| Associative final | 16 | 0.4250 | 0.7500 | 0.7500 | 0.3708 | 0.7500 |
| Associative final | 32 | 0.3875 | 0.7500 | 0.7500 | 0.3559 | 0.7500 |
| Direct final | 8 | 0.7500 | 0.7500 | 0.7500 | 0.3170 | 0.2500 |
| Direct final | 16 | 0.7500 | 0.7500 | 0.7500 | 0.2823 | 0.2500 |
| Direct final | 32 | 0.7500 | 0.7500 | 0.7500 | 0.2659 | 0.2500 |

These are observed-partition results, not independent validation. They cannot demonstrate generalization and are especially vulnerable to repeated exposure and dataset-level structure.

## Generalization conclusion

The 64/128 results are evidence of performance on larger collections and held-out scenarios under the declared split, but the limited eight-example training exposure materially weakens interpretation. A future independently generated development-evaluation corpus with sizes 8, 16 and 32, using new scenarios and new content identities, would help separate ordinary scenario generalization from the larger-collection scaling effect. It would not replace the final holdout.

No holdout was accessed. No training example was moved into validation.
