# LITE-007 Stage A Abstention Audit

Status: Gate A passed for the corrected development fixture; no optimizer updates were permitted until this audit completed.

The executable audit is `benchmarks/lite/lite007_abstention_audit.py`. It reruns lexical, semantic, and hybrid retrieval through the corrected `.encode()` path and records per-query eligibility, fallback, resolver validity, required evidence, and relevance-support status.

## Partition of the previously reported 43 expected abstentions

- Correct policy abstentions: 9.
  - Seven scenario-11 rows have no eligible public evidence for their required private evidence.
  - Two scenario-11 rows have no eligible collection under the requested scope.
- Remaining cases: 34.
  - 22 answerable `negation`/`superseded` rows were mislabeled as expected abstentions in the historical fixture. They have eligible supporting records and are not semantic insufficient-evidence cases.
  - 12 `missing` rows lack the requested battery evidence. They return resolver-valid but query-irrelevant authorized records instead of abstaining.

The 12 returns explain the previous “zero unsupported verified returns” result: the old check established source resolution and scope validity, not factual support for the query. They are now recorded as `verified_but_irrelevant_authorized_returns`.

The corrected audit reports separately:

- policy abstention;
- semantic insufficient-evidence abstention;
- fixture-label error;
- irrelevant authorized retrieval;
- resolver verification;
- fallback events;
- answerable-query false abstentions.

No blanket suppression threshold was added. The original fixture was not modified. The corrected labels will be represented in the independent LITE-007 corpus instead.

## Fallback integrity

The corrected semantic path invokes `.encode(query)`. The audit records `fallback_used` and `fallback_reason`. No semantic programming error was accepted as semantic success. In the audit run, lexical fallback events were zero; semantic no-compatible-candidate fallback events were recorded separately for policy-ineligible rows.
