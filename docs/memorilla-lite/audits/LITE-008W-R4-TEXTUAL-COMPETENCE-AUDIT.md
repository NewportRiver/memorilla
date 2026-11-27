# LITE-008W-R4 Textual Competence Audit

The R3 textual evaluation used the same frozen decoder candidate-scoring implementation for query-only, relevant evidence, counterfactual evidence, and unrelated evidence. Eligibility was a stricter composite criterion and is not equivalent to factual communication failure.

Metrics by partition:

```json
{
  "teacher_training": {
    "rows": 6,
    "eligible": 5,
    "relevant_correct": 6,
    "counterfactual_correct": 6,
    "complete_textual_reversals": 6,
    "unrelated_discrimination": 4,
    "query_greedy_correct": 4,
    "eligibility_rule": "abs(query_only margin)<=4; relevant margin>=1; counterfactual margin<=-1; unrelated margin>-1"
  },
  "calibration": {
    "rows": 2,
    "eligible": 2,
    "relevant_correct": 2,
    "counterfactual_correct": 2,
    "complete_textual_reversals": 2,
    "unrelated_discrimination": 2,
    "query_greedy_correct": 1,
    "eligibility_rule": "abs(query_only margin)<=4; relevant margin>=1; counterfactual margin<=-1; unrelated margin>-1"
  },
  "independent_development": {
    "rows": 8,
    "eligible": 3,
    "relevant_correct": 8,
    "counterfactual_correct": 8,
    "complete_textual_reversals": 8,
    "unrelated_discrimination": 3,
    "query_greedy_correct": 4,
    "eligibility_rule": "abs(query_only margin)<=4; relevant margin>=1; counterfactual margin<=-1; unrelated margin>-1"
  }
}
```

The audit distinguishes correct relevant directions, correct counterfactual directions, complete textual reversals, unrelated-evidence discrimination, query-only greedy correctness, and eligibility. Candidate preference reversals do not establish unrestricted generation. Precision and prompt-position differ between textual concatenation and latent-slot evaluation; no latent superiority claim is made.
