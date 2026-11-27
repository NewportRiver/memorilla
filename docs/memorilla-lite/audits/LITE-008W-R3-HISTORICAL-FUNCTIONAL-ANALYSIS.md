# LITE-008W-R3 Historical Functional Analysis

The four W-R2 leave-one-entity-out folds and full regularization schedule are preserved unchanged. R2's minimum-norm historical result is preliminary transfer evidence: three complete counterfactual reversals across four excluded entities, not independent acceptance.

The analysis separates residual reconstruction error, full embedding error, projection norm, latent norm, NLL, and candidate margin. Lower embedding MSE under stronger ridge does not guarantee factual behavior: ridge changes the latent direction and scale seen by the nonlinear frozen decoder, so decoder margins can worsen despite smaller Euclidean error. Functional behavior, not MSE alone, governs selection.

R3 selected λ=0.001 using historical and training evidence only; independent development data were not used for selection.
