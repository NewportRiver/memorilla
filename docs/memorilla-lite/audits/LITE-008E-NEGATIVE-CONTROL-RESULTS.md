# LITE-008E Negative Controls

All malformed-input controls were run against the native harness with the qualified Q4 model. Each failed safely with exit status `2` and an explicit validation error before native batch submission.

| Control | Result |
|---|---|
| Short embedding file | `embedding rows are too short` |
| NaN embedding value | `embedding contains non-finite value` |
| External index 99 for a five-token input | `external index out of range` |
| Context 5 for five inputs plus four continuation steps | `context overflow before native submission` |

No model, SQLite, persistent memory, neural checkpoint, or protected-data file was modified. Invalid artifact identity was not passed to the harness: the expected Q4 checksum was verified before execution. Unsupported token/embedding co-submission was avoided by sequential batches.

The native API's negative return path was also checked by the harness for batch addition, position setting, output-logit setting, and `llama_process`; no accepted primary run encountered one.
