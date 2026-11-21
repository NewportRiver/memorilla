# LITE-007M Scientific Disposition

## Status

LITE-007M is a documentation-only, read-only investigation. It performed no optimizer updates, created no replacement checkpoints, accessed no protected holdout, and made no architecture or dependency changes.

## Finding

LITE-007L established that broad deterministic exposure alone did not make the associative model competitive with the direct bilinear control on the independent development corpus. At 240 updates, associative Hit@1 was 0.1222 versus direct 0.8861; associative MRR was 0.2446 versus 0.9351; associative Complete@5 was 0.1333 versus 0.3361.

The associative model did learn: independent Hit@1 increased from 0.0222 at initialization to 0.1222 at 240 updates. However, it learned too slowly or into an insufficient representation for the declared objective.

## Root-cause confidence

The evidence supports a multi-factor explanation:

- High confidence: direct ranking has a simpler optimization path and a strong matched-exposure advantage.
- Medium confidence: associative slot routing and representation specialization are insufficient on the independent corpus.
- High confidence: the larger validation corpus is an inadequate sole basis for generalization because associative Complete@5 reaches a ceiling absent from the independent corpus.
- Medium confidence: collisions, missing abstention examples, and order/tie behavior are evaluation limitations.
- Low confidence: any single architectural component is proven to be the unique root cause.

## Scientific decision

Do not claim associative retrieval superiority. Do not reinterpret the larger-corpus Complete@5 ceiling as independent generalization. Preserve the LITE-007L negative result, direct-control advantage, ordering sensitivity of the zero control, and the initialization incident.

## Recommendation

The smallest scientifically responsible next step is read-only attribution and dataset-shortcut analysis using the existing checkpoints. If and only if that identifies a falsifiable mechanism, consider a separately authorized one-variable pilot capped at 40 updates per arm. Otherwise, prefer a simpler mobile-compatible retrieval architecture based on measured accuracy, latency, and RSS rather than extending the current associative architecture by default.

## LITE-008 gate

LITE-008 remains BLOCKED. No decoder conditioning, latent-token integration, or production claim is justified by LITE-007L or LITE-007M.

## Evidence and reproducibility

The committed LITE-007L artifacts are the source of numerical results. The hypothesis matrix records supporting and contradictory evidence, confidence, proposed minimal experiment, resource requirements, risks, and falsifiers for each ranked explanation. All proposed work preserves the existing Android resource constraints and excludes holdout data.
