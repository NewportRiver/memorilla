# LITE-008Q scientific acceptance contract

Predeclared thresholds before independent development scoring:
- reduced bridge-chain finite-difference maximum relative error <= 0.10;
- at least 6/8 entity-disjoint development counterfactual reversals;
- trained bridge mean counterfactual margin improvement >= 0.20 over untrained and memory-independent controls;
- relevant evidence must outperform mismatched evidence by >= 0.20 mean margin;
- finite gradients, frozen encoder/decoder, no nonfinite update, and resource limits;
- greedy output and NLL reported separately, not substituted for ranking.

Failure of any hard criterion prevents a generalized-bridge GO.
