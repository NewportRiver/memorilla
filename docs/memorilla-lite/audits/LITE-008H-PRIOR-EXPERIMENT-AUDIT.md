# LITE-008H Prior Experiment Audit

## Established findings

- LITE-008F established only 384-to-576 decoder-embedding reconstruction feasibility.
- LITE-008G corrected the MSE gradient defect and showed that corrected projection NLL improvements were also produced by zero, constant, and mismatched controls.
- LITE-008G relevant textual evidence improved NLL over query-only, so the decoder was capable of using explicit evidence.
- LITE-008G projected-memory counterfactual preference was positive in only 8/16 directions.
- LITE-008G greedy exact-answer accuracy was zero in the one-token setup.

## Unresolved hypotheses investigated here

- Projection parameter count may be too large relative to the small sample. H therefore uses a 16-coefficient fixed-basis diagnostic, not 221,760 unrestricted parameters.
- Constant and mismatched improvements may be global prompt/position effects rather than factual memory.
- The external slot changes causal positions; all latent conditions use one matched slot, while textual controls are reported separately.
- H uses one-token color answers for candidate ranking and preserves leading-space tokenization checks. Multi-token generation was not promoted after the prerequisite failure.
- Values are intentionally shared across partitions, but entity and fact identities are disjoint and the overlap is declared in the dataset manifest.
- The prior resource-monitor hang is diagnosed as captured native stderr pipe saturation; H redirects child streams and enforces timeout cleanup.

Historical F/G results are preserved unchanged.
