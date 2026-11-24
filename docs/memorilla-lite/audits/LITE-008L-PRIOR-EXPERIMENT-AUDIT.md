# LITE-008L Prior Experiment Audit

Baseline `fe0d6b3` and LITE-008K NO-GO were preserved.

Findings:
- Unrelated evidence in the K oracle selected another record's value without entity matching; this was corrected in the L textual evaluator.
- K counterfactual scoring did not independently verify that evidence named the queried entity and alternate candidate.
- EOS-derived embeddings were included in textual controls rather than treating textual and latent controls as separate conditions.
- Genuine and initial controls were duplicated in the K implementation.
- Three updates were insufficiently justified.
- K ran before the numerical prerequisite was fully resolved.
- K interrupted attempts did not have durable update counts; L records unknown counts as unknown and does not reconstruct them.

Historical K artifacts were not overwritten. L uses a new dataset, new ledger, and new result files.
