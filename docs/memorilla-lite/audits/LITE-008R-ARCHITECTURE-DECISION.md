# LITE-008R architecture decision

Retain `Linear(384,576)` as the baseline. Do not add layers or increase training duration. The smallest justified change is numerical qualification and a fail-closed gate. Stabilization candidates remain unexecuted because the prerequisite failed.

Residual projection, norm regularization, bounded residual parameterization and feature normalization are not selected yet. Selecting among them before a valid gradient chain would confound mechanism repair with optimization.
