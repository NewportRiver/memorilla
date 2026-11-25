# LITE-008R scientific acceptance contract

Predeclared gates:
- Stage A-D qualification artifacts must all be `pass`; relative error threshold `0.10`; Stage C cosine threshold `0.999`; Stage B absolute error threshold `1e-3`.
- No scientific bridge optimizer update is allowed before qualification.
- Stable learning must remain below the 1 GiB training RSS ceiling, stop on nonfinite values or declared norm instability, and report displacement and optimizer state.
- Fresh entity-disjoint development requires at least eight counterfactual pairs, six of eight reversals, and improvement over untrained, zero, mismatched, soft-prompt and scale-matched random controls.

The contract is fail-closed. R did not pass the numerical prerequisite, so all scientific training and independent acceptance are blocked.
