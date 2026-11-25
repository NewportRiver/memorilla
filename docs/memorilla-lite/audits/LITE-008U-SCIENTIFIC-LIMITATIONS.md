# LITE-008U limitations

Teacher generation succeeded for 4 pairs and 8/8 assigned factual preferences, but teacher embeddings were generated with individually optimized latent slots and have norms 6.45–8.05, above ordinary EOS scale. Teacher geometry is based on only eight samples. Exact per-update gradient cosine was unavailable because LITE-008T did not persist gradient vectors.

Student alignment achieved low embedding MSE and high cosine, yet reproduced 0/4 counterfactual reversals on its own training cases. This blocks independent development evaluation and decoder-loss refinement. No causal claim about capacity, duration or interference is justified.

Soft-prompt, ordinary-text fresh control, independent development, refined student, query-conditioned comparative student, nonlinear student and protected holdout phases were unexecuted.
