# LITE-007F Report — Native Associative Memory Projection Learning

Status: projection-only associative-memory pilot passed all mathematical and resource gates. LITE-007 remains incomplete.

## Gradient implementation

The exact authorized tensors were `mem_k_w`, `mem_k_b`, `mem_v_w`, and `mem_v_b`. NumPy reverse-mode propagation now covers masked summary initialization, query-conditioned slots, layer normalization, scaled multi-head attention, masked softmax, attention aggregation, residual paths, GELU feed-forward refinement, and the corrected pairwise evidence-ranking objective. Frozen downstream tensors propagate gradients but are not updated.

No autodiff framework, full Jacobian, or new dependency was added.

## Verification

Finite differences across three seeds and 2/4/6-record collections passed:

- Maximum absolute error: `3.2231e-05`
- Acceptance tolerance: `1e-03`

Isolated PyTorch autograd comparison passed:

- Loss absolute error: `0`
- Maximum projection-gradient absolute error: `5.006e-09`

Masked/padded records, multiple positives, hard negatives, empty eligibility, nonfinite rejection, deterministic execution, and small-step descent were covered. Targeted tests: `13 passed in 0.86s`.

## Pilot

- Updates: `8`
- Trainable parameters: `mem_k_w`, `mem_k_b`, `mem_v_w`, `mem_v_b`
- Scoring head: frozen
- All other architecture tensors: frozen
- Initial diagnostic loss: `0.6864654421806335`
- Final diagnostic loss: `0.6862652897834778`
- Loss change: `-0.00020015239715576172`
- Peak RSS: `69,693,440` bytes
- Duration: `0.538305117` seconds
- Frozen checksum verification: passed
- Numerical/resource guards: passed

The controlled diagnostic loss improved. This is optimization evidence only, not evidence of cognition or generalization.

## Representation analysis

On the fixed diagnostic query:

- Latent L2 delta: `0.0326754`
- Relative latent delta: `0.00148576`
- Mean slot cosine: `0.9999989`
- Attention L1 delta: `4.53e-06`
- Slot pairwise similarity changed from `0.5110789` to `0.5110883`

The representation changed slightly and only the four authorized projection tensors changed. This does not establish improved associative reasoning.

## Paired controls

Validation evaluation used 175 answerable and 65 abstention examples with identical eligibility and evidence budgets:

- Untrained architecture loss: `0.7003155`, MRR `0.6153`
- Scoring-head trained loss: `0.7001796`, MRR `0.6153`
- Projection trained loss: `0.7002520`, MRR `0.6144`
- Both trained loss: `0.7001167`, MRR `0.6125`
- Direct query/evidence cosine MRR: `0.7792`

The direct cosine control was stronger on this synthetic fixture. No generalization improvement is claimed.

## Checkpoints and provenance

The successful LITE-007E scoring-head checkpoint was preserved and reloaded. The LITE-007F projection checkpoint was written atomically, is non-executable, passed encoder compatibility and checksum validation, and reproduced inference after reload. Both generated weight artifacts remain outside Git. The invalidated early LITE-007E pilot remains documented separately and produced no published weights.

The prepared train artifact remains bound to the development source and encoder identities. A separate validation artifact was created from the same authorized development corpus. No holdout labels or payloads were accessed or generated.

## Regression and blockers

Targeted gradient, optimizer, loader, objective, and integrity tests passed: `13 passed in 0.86s`. The complete inherited LITE regression remains incomplete because upstream fixtures require unavailable `transformers`; no installation or silent exclusion was performed.

Remaining scientific blockers:

- No full associative-memory training
- No memory-projection checkpoint selection
- No holdout evaluation
- No Android device inference acceptance
- No evidence of cognitive or generalization improvement
- Direct retrieval controls remain stronger on the current synthetic validation fixture

Stop here. No additional neural component was trained.
