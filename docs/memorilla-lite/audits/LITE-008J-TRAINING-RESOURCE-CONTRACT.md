# LITE-008J Training Resource Contract

Owner-authorized temporary training envelope:

- Maximum training RSS: `1,073,741,824` bytes
- Minimum system available-memory reserve: `536,870,912` bytes
- CPU-only, one numerical thread, one computational subprocess
- Batch size 1
- Sequence length at most 32
- One external memory slot
- Per-process timeout 300 seconds
- Cumulative experiment limit 30 minutes
- Maximum embedding updates 32
- Decoder frozen; no decoder optimizer state

The existing 256 MiB ceiling remains the inference ceiling and was retained for the separate Q4 transfer checks. The training monitor used sampled RSS/PSS and `/proc/meminfo`; it terminated on observed ceiling or reserve violation. This is enforceable sampled monitoring, not a kernel hard-memory guarantee.

No Android swap configuration was changed and no unrelated process was terminated.
