# LITE-008S corrective implementation

`experiments/lite008s/qualification.py` implements measured A-D status, Float64 synthetic checks, corrected metrics, detached positional state, forward-only finite differences, multiple external inputs, reduced depths and full-depth bridge parameter directions. `diagnose.py` records input identity, aliasing, determinism and directional checks. `bridge_train.py` requires the S identity-complete artifact and validates it inside `train_bridge()` before optimizer construction. No training ran.
