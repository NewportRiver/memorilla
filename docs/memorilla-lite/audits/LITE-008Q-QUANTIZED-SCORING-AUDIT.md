# LITE-008Q quantized scoring audit

The existing P harness accepted a target-token file and was previously used with two candidate IDs in one invocation. Q does not treat those outputs as an independent A/B margin. Corrected scoring runs candidate A and candidate B separately with the identical query IDs, identical external embedding bytes (`float32`, little-endian, `[sequence,576]`), identical memory position zero, fresh process/KV state, one target token, and a separate answer-position log probability.

Tokenizer compatibility and one-token candidate identities were checked with the pinned SmolLM2 tokenizer. Q4 inference was monitored with nonblocking stdout/stderr redirection and `/proc` RSS/PSS sampling.
