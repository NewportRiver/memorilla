# Functional Distillation Proposal

R2 establishes that embedding-space reconstruction can preserve historical decoder behavior when the input feature rows are exactly interpolated. It does not establish that MSE is a sufficient future learning objective.

Recommended future qualification order, with no implementation authorized here:

1. Residual embedding reconstruction against qualified teacher vectors.
2. Correct-answer likelihood and counterfactual margin matching on the exact frozen decoder.
3. Candidate-logit matching, using independently evaluated A/B candidate states.
4. Combined loss only after the exact student objective receives numerical qualification.

The reconstruction result suggests decoder functional supervision is a useful auxiliary target when feature-to-teacher interpolation is imperfect. It does not authorize decoder-dependent optimization, new students, or generalization claims.
