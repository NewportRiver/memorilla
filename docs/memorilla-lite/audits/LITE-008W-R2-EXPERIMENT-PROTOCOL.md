# LITE-008W-R2 Experiment Protocol

All scientific subprocesses use `experiments/lite008w/run_qualified.sh`, which invokes the LITE-008D-R4 interpreter and fail-closed preflight. Resource limits remain unchanged: CPU-only, one thread, one subprocess, 1 GiB scientific ceiling, 512 MiB minimum available memory, sequence length 32, one slot, 300-second individual subprocess maximum, 30-minute cumulative maximum, and 256 optimizer updates.

Historical functional cross-validation and textual competence were completed. Fresh teacher generation was attempted with the authorized six-pair/192-update schedule but was interrupted after exceeding the 300-second individual subprocess limit. Its partial ledger is retained; no incomplete teacher artifact is valid.
