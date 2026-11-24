# LITE-008K Experiment Protocol

## Predeclared gates

1. Verify artifact identities and immutable MiniLM features.
2. Correct numerical dtype boundaries and answer tokenization.
3. Establish stable-enough multi-direction numerical gradient evidence.
4. Pass a four-pair counterfactual oracle: every A embedding prefers A and every B embedding prefers B.
5. Only then train a compact generalized bridge.

The protocol stopped at gate 4. The oracle result was `1/4`, below the required `4/4`.

## Budget

- Training RSS ceiling: 1 GiB
- Available-memory reserve: 512 MiB
- One CPU thread and subprocess
- Sequence length ≤32
- One slot
- Individual process ≤300 s
- Cumulative execution ≤30 min
- Maximum provisional bridge updates: 256

The persisted successful oracle used 24 optimizer updates. A preceding timed-out implementation had an intended 48-update budget but did not persist a phase ledger; its exact completed update count is unavailable. The conservative attempted upper bound remained below 256. No generalized bridge update was started.

## Controls

Textual calibration used query-only, relevant textual evidence, unrelated textual evidence, and counterfactual textual evidence. The latent bridge, development evaluation, quantized transfer, and complete control matrix were not run after the oracle failure.
