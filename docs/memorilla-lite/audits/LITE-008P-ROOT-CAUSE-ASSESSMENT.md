# LITE-008P Root-Cause Assessment

## Supported findings

1. Insufficient duration was a real contributor. O stopped at update 8. P calibration trajectories continued changing materially through update 16; both calibration pairs reached the intended A/B signs under the selected configuration.
2. The objective contains competing components. NLL and margin-only gradients had cosine `0.4923`, so the contrastive objective is not equivalent to a single aligned NLL descent direction.
3. The one-slot representation is not proven insufficient. A fresh four-pair oracle passed with one slot after 16 updates.
4. Additional slots were not beneficial in the bounded four-update pilot; the pilot was too short to establish a capacity limit.
5. A multi-slot gradient-shape defect was found and corrected before P scientific updates. The representative two-slot finite-difference relative error was `0.000407` after correction.

## Not established

The data do not distinguish whether a different optimizer, longer than 16 updates, or a larger latent capacity would further improve performance. The selected configuration was not an exhaustive hyperparameter search. The result does not establish generalized semantic memory.
