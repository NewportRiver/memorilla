# LITE-008U protocol

Baseline: `cd6760252e55b21dd03483790c5d7b45428a87c8`. CPU-only, one numerical thread, one subprocess, sequence length <=32, one slot, training RSS <=1,073,741,824 bytes, inference RSS <=268,435,456 bytes, reserve >=536,870,912 bytes, cumulative U optimizer cap 256 and wall cap 30 minutes.

Eligibility was fixed by ordinary-text query-only/relevant/counterfactual/unrelated scoring before latent optimization. Four Q train cases (00–03) were selected because the textual prerequisite accepted them. Each A/B teacher received identical EOS initialization, the same query, frozen decoder, 16 Adam updates at LR 0.05, and no more than 32 permitted updates.

Student architecture was fixed before alignment: `z = EOS + W*x`, W shape 576x384, zero initialized, frozen EOS, frozen MiniLM, MSE to teacher embeddings, paired A/B ordering. Student received 128 updates at LR 0.001. This exhausted the U budget after 128 actual teacher updates. Decoder refinement and independent evaluation were therefore precluded after the training-semantic gate failed.
