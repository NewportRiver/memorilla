# LITE-008K Learning Objective

The diagnostic objective was:

`L = NLL(A) + 0.5 * softplus(-margin)`

where:

`margin = log P(A | Q, M) - log P(B | Q, M)`

The counterfactual direction swaps A and B while retaining the same query. Decoder logits are cast to Float32 before `log_softmax`; the external embedding remains a Float32 master parameter and is cast to BFloat16 only at decoder input.

For correct-class log-probabilities `l_A` and `l_B`, the margin contribution has gradient:

`d softplus(-m) / d m = -sigmoid(-m)`

and `dm = dl_A - dl_B`. Thus the contrastive term increases the correct-vs-alternate margin while the NLL term preserves answer likelihood pressure.

The implementation is isolated in `experiments/lite008k/objective.py` and regression-tested. It was used for the oracle diagnostic only. No generalized bridge optimizer was started because the mandatory oracle prerequisite failed.
