# LITE-008G Decoder Scoring Contract

The native LITE-008E harness now accepts `--targets target_ids.bin`. In scoring mode it:

1. Processes the prompt using token, embedding, or mixed input according to the existing harness mode.
2. Reads logits at the causal position after the prompt.
3. Computes stable log-softmax using max subtraction and double-precision accumulation.
4. Records the target token log-probability.
5. Submits the target token as the next causal token.
6. Repeats until the target sequence is exhausted.
7. Emits `target_tokens`, `target_logprobs`, `total_log_likelihood`, and `mean_nll`.

Greedy generation remains a separate run, because teacher-forcing target tokens and greedy tokens produce different causal histories. The evaluation runner records both outputs.

## Equivalence validation

For prompt `The capital of France is` and target ` Paris`, ordinary token input and equivalent decoder embedding input produced identical results:

- target token: `7042`
- total log-likelihood: `-2.04166`
- mean NLL: `2.04166`
- both native runs exited 0

Target values in the development dataset were verified to tokenize as exactly one token with a leading space. Prompt boundaries and latent-slot positions were held constant for latent conditions; textual evidence comparisons are reported separately because their full prompt lengths differ.
