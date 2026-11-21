# LITE-007M Corrective Experiment Proposal

## Recommendation

Do not authorize another full comparative training run yet. The smallest next step is a read-only diagnostic package using the existing LITE-007L checkpoints, followed only by a one-variable bounded training experiment if the diagnostics identify a specific bottleneck.

## Stage 0: read-only diagnostics

Use no optimizer updates and create no checkpoints. Analyze:

1. Per-example and per-collection-size score margins for associative, direct, and cosine models.
2. Required versus nonrequired evidence score distributions.
3. Gradient/Jacobian attribution from the pairwise loss to each associative ranking-connected tensor using existing code paths, without updating parameters.
4. Slot assignment entropy, slot diversity, query sensitivity, and evidence-order stability stratified by category and required-evidence count.
5. Identity/template overlap and construction differences between training, independent development, and larger development-validation data.
6. Results with collision groups reported separately and excluded as a sensitivity analysis.

Success is diagnostic specificity, not a higher metric. The output must identify whether the dominant loss is before the slot representation, in the score head, in gradient attenuation, or in data/evaluation construction.

## Stage 1: smallest possible authorized training comparison

Only if Stage 0 produces a concrete prediction, authorize one variable at a time:

- Preserve the `lite006-associative-v1` architecture.
- Preserve the same independent development corpus and matched schedule.
- Preserve FP32, one thread, one subprocess, RSS <= 256 MiB.
- Use a small predeclared pilot budget, no more than 40 updates per arm.
- Change exactly one factor: optimizer conditioning, learning-rate schedule, or score-path treatment—not multiple factors together.
- Keep the direct and cosine controls read-only unless a separate control is explicitly authorized.
- Evaluate at 0, 20, and 40 updates with Hit@1, Recall@1/5, MRR, Complete@5, margins, and representation diagnostics.

A pilot succeeds only if it improves the predeclared associative independent-corpus endpoint without exceeding resource limits and without degrading the direct-control comparison through an unplanned metric change. It fails if improvement is absent, only appears on the larger validation corpus, depends on collision/order strata, or violates the resource guard.

## Alternative architecture decision gate

If Stage 0 shows that associative gradients reach the score path but evidence-selective representations remain weak, do not automatically expand the architecture. Compare against simpler mobile-compatible retrieval candidates, such as conventional cosine retrieval with deterministic evidence diversification or a compact direct ranker, under matched data and evaluation. The simpler candidate is preferable if it provides materially better independent-corpus accuracy at equal or lower RSS and latency.

## Resource and safety requirements

- No GPU, cloud inference, external provider, large dependency, or additional memory database.
- Preserve the 256 MiB RSS ceiling, one numerical thread, one computational subprocess, and existing RAM/storage guards.
- No final holdout access.
- No automatic retries or budget transfer.
- No generated weight publication.
- LITE-008 remains blocked until a separately reviewed scientific result exists.

## Falsification and stopping

Stop the diagnostic program if evidence identities or checkpoint hashes do not match the LITE-007L manifests. Stop any pilot on resource breach, nonfinite values, parameter corruption, sample repetition, or failure to improve the predeclared independent endpoint. A negative pilot is a valid result and should reduce confidence in the proposed corrective mechanism.
