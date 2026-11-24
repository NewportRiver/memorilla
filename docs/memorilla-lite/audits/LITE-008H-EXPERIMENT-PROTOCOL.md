# LITE-008H Experiment Protocol

## Predeclared limits

- Maximum total optimization budget: 256 decoder evaluations
- CPU-only, one numerical thread, one native subprocess
- Decoder RSS ceiling: 268,435,456 bytes
- Maximum wall duration: 30 minutes
- Oracle coefficients: 16
- SPSA iterations: 8
- Perturbation: `0.05`
- SPSA learning rate: `0.4`
- Random-search control: 8 vectors
- Seed: `20261124`
- Protected holdout: inaccessible

## Acceptance criteria

Oracle feasibility requires optimized correct-vs-competing margin to exceed initial, exceed budget-matched random search, and show reliable memory-dependent counterfactual behavior. Generalized GO additionally requires development improvement over untrained and memory-independent controls, correct > mismatched, consistent counterfactual reversal, and answer-generation evidence.

## Stop rule

If the decoder cannot exhibit reliable memory-dependent conditioning under the 16-coefficient oracle diagnostic, do not optimize the full bridge. That stop rule was reached.
