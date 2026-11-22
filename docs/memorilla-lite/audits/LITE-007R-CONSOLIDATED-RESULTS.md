# LITE-007R Consolidated Results

## Scope and evidence boundary

This closeout consolidates LITE-007J through LITE-007Q. It preserves negative, inconclusive, and control results. No new training, architecture implementation, checkpoint replacement, holdout access, external inference, APEX integration, or LITE-008 implementation occurred.

The principal independently evaluated associative result is LITE-007L: 240 associative updates versus 240 direct-bilinear updates, on the matched 360-query independent development corpus.

## Retrieval comparison

| Method | Independent Hit@1 | MRR | Complete@5 | Evidence status |
|---|---:|---:|---:|---|
| Associative memory, LITE-007L final | 0.1222 | 0.2446 | 0.1333 | Independently evaluated; negative versus direct control |
| Direct bilinear ranking, LITE-007L final | 0.8861 | 0.9351 | 0.3361 | Independently evaluated; strongest learned ranker |
| Direct cosine, LITE-007J conventional artifact | unavailable as identical LITE-007L endpoint | see size-stratified J artifact | size-stratified: 0.225–0.4917 | Combined development denominator; not relabeled independent |
| Existing hybrid retrieval, LITE-006 corrected artifact | unavailable as identical LITE-007L endpoint | unavailable | required-only completeness 1.0 on 77 required-evidence queries | Conventional artifact; different corpus/protocol |

The LITE-007J conventional comparison reported direct cosine Complete@5 from `0.4917` at collection size 8 to `0.2250` at 128, and hybrid from `0.4917` to `0.2000`. Those are useful historical controls but are not interchangeable with LITE-007L's independent answerable denominator.

The LITE-006 corrected conventional benchmark reported required-only recall and complete support of `1.0` for lexical, semantic, and hybrid retrieval over 77 required-evidence queries. It also reported mean complete-pipeline latency of `0.467 ms` lexical, `5.013 ms` semantic, and `5.443 ms` hybrid. The benchmark's abstention-policy accuracy was `9/43`; this is preserved rather than silently converted into retrieval accuracy.

## What the sequence established

- LITE-007J established pretraining controls and showed that an attention/FFN associative variant was not a substitute for a conventional control.
- LITE-007K qualified the independent corpus and checkpoint/evaluation identities.
- LITE-007L supplied the decisive corrected comparative training result: direct bilinear ranking substantially exceeded the associative model on independent development retrieval.
- LITE-007M documented plausible causes without claiming that one cause was proven.
- LITE-007N found negative associative margins, nearly uniform evidence attention, weak required-evidence enrichment, and dataset-template collision limitations.
- LITE-007O found that query/key gradient attenuation persisted after parameter-scale normalization, while upstream finite-difference validity remained provisional.
- LITE-007P qualified the native reverse-mode implementation under a bounded FP32 fixture and classified small-epsilon discrepancies as finite-difference conditioning, not a verified code defect.
- LITE-007Q implemented a non-trainable attention-logit scale with exact scale-1 compatibility. Frozen scales `0.5`, `1`, `2`, and `4` changed logit spread but all produced Hit@1 `0.1222`; no retrieval benefit was observed.

## Operational conclusion

Conventional retrieval remains the operational baseline. It is persistent, inspectable, Android-compatible, and has the strongest directly measured evidence for the product objective. The failed associative ranking result does not prove latent memory conditioning impossible; it shows that this compact associative ranker, training setup, and evidence-routing behavior did not justify replacing conventional retrieval.
