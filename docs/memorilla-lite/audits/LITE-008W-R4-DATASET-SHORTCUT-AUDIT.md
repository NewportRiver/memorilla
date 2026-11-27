# LITE-008W-R4 Dataset Shortcut Audit

Frozen independent development inventory: 8 entities. Candidate answer vocabulary counts: {'linen': 3, 'amber': 5, 'marble': 5, 'cedar': 3}. Entity-prefix counts: {'entity': 8}. Candidate assignment counts: {('linen', 'amber'): 1, ('amber', 'marble'): 1, ('marble', 'linen'): 1, ('cedar', 'amber'): 1, ('linen', 'marble'): 1, ('amber', 'cedar'): 1, ('marble', 'amber'): 1, ('cedar', 'marble'): 1}. All development rows use the repeated synthetic record template with distinct entity suffixes and a small answer vocabulary.

The projection therefore demonstrates bounded entity generalization within a shared answer vocabulary, not unseen-vocabulary semantic generalization. Candidate-position and vocabulary shortcuts cannot be ruled out by the eight reversals.

The mechanism diagnostics tested full relevant evidence, answer-value-only evidence, same-answer other-entity evidence, entity-only evidence, opposite factual value, meaning-preserving paraphrase, and conflicting evidence. Results are in `LITE-008W-R4-SEMANTIC-MECHANISM-DIAGNOSTICS.json`; they are exploratory, forward-only, and were not used for fitting or selection.
