# LITE-008W-R3 Interruption Root Cause

W-R2 used one all-or-nothing in-memory teacher-generation process. It appended durable update ledger records but delayed teacher-vector publication until all six pairs completed. The process exceeded the 300-second individual limit and was terminated.

Audit of the retained ledger found 128 entries, not the 126 reported in R2: four complete pairs, 16 updates per direction. No teacher vectors or optimizer states from the interrupted R2 process were recoverable. R3 preserves that ledger and charges all 128 updates.

R3 replaces this with one-direction work units, per-update checkpoints, atomic final artifacts, reload verification, and a durable completion manifest.
