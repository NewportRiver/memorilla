# LITE-008H Optimization Methods

## SPSA

Selected for the diagnostic because it uses two simultaneous perturbation evaluations per objective and does not require decoder gradients. H used a fixed 16-dimensional coefficient vector, a fixed normalized random basis `[16,576]`, perturbation `epsilon=0.05`, learning rate `0.4`, eight iterations, and paired correct/alternate target scoring.

## Low-dimensional derivative-free search

The fixed-basis parameterization is the bounded low-dimensional search space. It has 16 trainable coefficients and can express only a 16-dimensional subspace of one continuous decoder slot. It is not a deployable bridge and cannot represent a general semantic mapping by itself.

## Random-search control

Eight random coefficient vectors used the same correct-vs-alternate margin objective and equivalent diagnostic budget. The best random margin exceeded the SPSA result.

## Differentiable input gradients

Not executed. The pinned native runtime remains inference-only and no compatible local decoder differentiation path exists without introducing a new decoder/training runtime, which is prohibited.

## Larger optimization

Not executed. The oracle prerequisite failed, so unrestricted bridge optimization and memory-independent soft-prompt optimization were stopped before spending the remaining budget.
