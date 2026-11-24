# LITE-008M Scientific Limitations

- Full FP32 forward/backward qualification was not established; the reference exceeded the authorized RSS ceiling.
- The reduced component excludes all transformer attention and feed-forward layers. Its strong BF16/FP32 agreement cannot be generalized to the complete decoder.
- The corrected textual dataset is synthetic. It exposes the LITE-008L answer-frequency defect but is not a protected factual holdout.
- Textual competence was evaluated by candidate ranking, not unrestricted generation; generation accuracy was not used to qualify latent learning.
- The first M textual run timed out because the evaluator used two passes for every case. It produced no aggregate result; incremental ledger case records were retained. The corrected evaluator completed.
- A reduced component bookkeeping defect and an invocation/import failure were preserved as operational history, not scientific results.
- No oracle optimization, generalized bridge training, decoder fine-tuning, alternative model, or associative-memory training occurred.
