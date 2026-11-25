# LITE-008Q prior experiment audit

Baseline `89c3cc8`; LITE-008P preserved the O failure and passed isolated FP32 optimization 4/4 on fresh pairs. P used entity-disjoint calibration/oracle partitions, contrastive objective, one EOS-initialized slot, and 224 optimizer updates. P recovered optimized slots from durable JSON and measured candidate NLL, margins, greedy IDs, norms and textual controls.

The P Q4 wrapper supplied `[A,B]` to a sequential target interface; this did not establish an independent same-position margin. Q corrected this with independent one-token runs. P’s multi-slot correction changed grouped hidden-state extraction to remove the batch dimension (`gh[0,:n,:]`).

Q retains the distinction between P’s update-duration evidence and dataset changes: P’s longer trajectories and fresh entities changed both factors; eight versus sixteen updates is not a universal convergence threshold.
