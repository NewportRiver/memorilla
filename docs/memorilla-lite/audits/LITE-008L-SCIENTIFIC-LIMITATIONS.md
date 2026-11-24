# LITE-008L Scientific Limitations

- The textual dataset is synthetic and small: four calibration rows and eight development rows.
- Textual competence is candidate ranking, not unrestricted generation or factual world knowledge.
- The selected prompt format tied with the other formats on calibration; the predeclared tie rule selected question-answer.
- No protected holdout was accessed.
- The Float32 reference was blocked by the existing RSS ceiling before a forward/backward result.
- BF16 finite differences characterize a rounded computation and should not be expected to equal the continuous master-parameter derivative at arbitrary epsilon.
- No oracle requalification, generalized bridge training, Q4 transfer, or associative-memory training occurred.
- Resource monitor invocations missing `PYTHONPATH` are operational failures, not scientific results; they are preserved as ledger/telemetry history.
