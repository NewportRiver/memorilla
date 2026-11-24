# LITE-008I Prior Experiment Corrections

LITE-008F, G, and H remain valid historical evidence.

Corrections and boundaries:

- H was a one-example SPSA oracle diagnostic, not a general optimizer evaluation.
- H used a random 16-dimensional basis, so it did not test the full 221,760-parameter projection.
- SPSA and random search did not have identical candidate counts despite the intended budget-matched comparison; this weakens optimizer comparison but does not change SPSA's failure to beat its own initial margin.
- H's textual irrelevant and counterfactual conditions used the same alternate value; they were distinct semantic conditions but not distinct distractor strings.
- H used one-token candidate answers and one-step greedy generation, so it did not qualify unrestricted multi-token answer generation.
- H did not provide differentiable decoder feedback.

These limitations prevent interpreting H as a test of the original differentiable Memorilla mechanism. They do not establish that the latent semantic-memory hypothesis is impossible.
