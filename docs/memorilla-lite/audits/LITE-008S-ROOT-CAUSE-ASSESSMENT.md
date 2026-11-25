# LITE-008S root cause

The exact defect was a graph mismatch in the R conventional reference: rotary/position construction received a bridge-connected hidden tensor, while manual recomputation detached hidden first. The unintended autograd path was not present in manual VJP. After correction, forward-only and differentiable losses agree and all tested external inputs pass reduced VJP checks.
