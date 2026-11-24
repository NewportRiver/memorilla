# LITE-008L Experiment Ledger Contract

`experiments/lite008l/ledger.py` appends fsynced JSONL records under `~/.cache/memorilla-lite/LITE-008L/experiment-ledger.jsonl`.

Each record includes event ID, timestamp, experiment, phase, and status. Phase records additionally retain exact seed/configuration, case identity, forward counts, resource outcomes, termination reason, and completed/unknown update accounting. Case records are persisted immediately after each completed decoder case. No interrupted optimizer update count is reconstructed.

LITE-008L used no optimizer updates. The Float32 reference was terminated before forward/backward completion after crossing the RSS ceiling; its attempted and completed update counts are explicitly zero.
