# LITE-007O Hypothesis Decision

## Verified scope

LITE-007O was run from baseline `75e9346` on `feature/memorilla-lite-android`. It used the frozen LITE-007L final associative checkpoint, the frozen initial associative checkpoint, and the same prepared independent development queries/evidence used by LITE-007N.

No optimizer update, checkpoint write, dependency installation, holdout access, or architecture change occurred.

## Hypothesis table

| Hypothesis | Evidence | Contradictory evidence / limitation | Decision |
|---|---|---|---|
| True attention-path gradient attenuation | After parameter-scale normalization, `b0.q_w` gradient RMS / parameter RMS is `2.28e-6`; `b0.k_w` is `4.01e-6`. Their proposed FP32 update ratios are `2.28e-9` and `4.01e-9`. Final scoring ratios are approximately `2.19e-5` (`score_slot_w`) and `3.81e-5` (`score_mem_w`). | Raw and normalized ratios are parameterization-dependent. Upstream finite-difference agreement is weaker for `query_w` and `mem_v_w`. | Supported, medium-high confidence as a measurable path bottleneck; not proven as the sole causal root. |
| Raw-gradient difference is only tensor-scale/dimensionality artifact | All 27 tensors were normalized by parameter RMS and dimensionality was recorded. The block attention query/key ratios remain orders of magnitude below score and value paths. | RMS normalization does not remove all conditioning or Jacobian effects. | Contradicted as a complete explanation. |
| Attention softmax saturation / near-uniform routing | Attention entropy is approximately `ln(N)` for each eligible position count: `2.07942` for 8, `2.77256` for 16, and `3.46571` for 32. Mean attention-logit standard deviation is only about `0.007`. Initial and final entropy/logit spread are nearly identical. | The analysis is frozen-checkpoint attribution, not a causal intervention. | Supported, high confidence as a routing-state observation. |
| Insufficient evidence-selective routing | Required-evidence enrichment remains approximately uniform in LITE-007N (`0.99876`), and initial/final entropy does not materially change. | Query conditioning and nonzero gradients exist; routing is not completely insensitive. | Supported, high confidence. |
| Gradient implementation fully validates every upstream attribution | Centered finite differences validate slots and final scoring well, but upstream query/key/value agreement is weaker; `query_w` is especially discrepant. | Float32 and nonlinear normalization complicate finite differences. | Unresolved; upstream analytical attribution must not be overclaimed. |
| Dataset construction alone explains the routing failure | Independent-corpus collision participation is widespread, and collision exclusion has zero denominator. | The normalized routing bottleneck persists as a model-side observation; no holdout was accessed. | Supported as a confounder/limitation, not isolated as the mechanism. |

## Corrective decision

One measurable model-side mechanism is supported: attention logits remain almost uniform and the normalized update opportunity for block query/key projections is extremely small. If an additional experiment is authorized, the smallest single variable is an attention-logit temperature or equivalent scale intervention, with all other architecture, optimizer, corpus, scheduler, and resource settings fixed.

That experiment is not authorized by LITE-007O. Until it is separately approved and its finite-difference/gradient contract is repaired for upstream projections, conventional retrieval remains the safer Android choice. No broad associative retraining or LITE-008 implementation is justified.

## Falsification criteria

The attention-path hypothesis should be rejected if a scale-normalized, implementation-validated one-variable pilot produces no measurable increase in attention-logit spread, required-evidence enrichment, independent-corpus margin, or Hit@1; or if corrected Jacobian attribution shows upstream query/key sensitivity comparable to final scoring after accounting for parameter RMS.
