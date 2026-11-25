# LITE-008Q experiment protocol

Architecture: shared `Linear(384,576)`; Float32 master parameters; EOS-bias/small-weight initialization; Adam learning rate 0.05; gradient clipping 1.0; batch size 1; one slot; one numerical thread; frozen MiniLM and decoder. Primary loss: answer NLL + 0.5 softplus(-signed correct margin). Matched NLL-only comparison and memory-independent soft prompt each used 16 updates. Primary bridge used 48 updates over 16 train examples repeated for three epochs.

Dataset: 8 train, 4 calibration, 8 independent development entities; randomized unfamiliar identifiers; three evidence templates; two query templates; balanced counterfactual assignments; answer-bearing evidence; no protected holdout. Configuration was frozen before development scoring. No development embedding was optimized.

Optimizer ledger: 160 successful Q bridge/control updates; one failed pre-update shape attempt and failed diagnostic harness attempts are recorded as non-scientific attempts and not counted as updates.
