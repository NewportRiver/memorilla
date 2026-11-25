# LITE-008V-R2 Resource Classification

This milestone uses the authorized scientific reference evaluation class, not production inference.

- Production inference RSS ceiling: 268,435,456 bytes; unchanged and not qualified by this run.
- Scientific reference evaluation RSS ceiling: 1,073,741,824 bytes; used for this isolated forward-only run.
- Experimental training ceiling: 1,073,741,824 bytes; no training was performed.
- CPU-only, one numerical thread, one subprocess, batch 1, sequence length <=32, one latent slot, reserve >=536,870,912 bytes, subprocess <=300 seconds.
- Scientific optimizer updates: zero.

The existing P frozen-decoder implementation was reused. No layer-streaming architecture, Q4 pathway, production deployment or limit change was introduced.
