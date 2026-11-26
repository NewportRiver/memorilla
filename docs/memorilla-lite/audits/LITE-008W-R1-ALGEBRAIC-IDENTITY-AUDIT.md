# LITE-008W-R1 Algebraic Identity Audit

`experiments/lite008w/analyze.py` now binds teacher and feature records by `(case_id, target)` keys and fails closed if the key sets differ or contain fewer than eight records. It no longer treats adjacent row position as the association contract.

The reconstruction bundle is verified against its retained SHA-256 before loading. Residual targets are computed as teacher embedding minus EOS. Algebraic metrics remain residual-space metrics; full embedding norms are evaluated separately by adding EOS before norm calculation in the runtime qualification path.

Historical W reports are preserved. The corrected implementation is committed as a fail-closed audit improvement.
