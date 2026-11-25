# LITE-008T graph root-cause verification

The original R reference path and corrected S path were compared conceptually and through the retained deterministic reproduction. The affected boundary is rotary/position-state construction. The R conventional path received a bridge-connected hidden tensor; the manual path detached hidden before constructing positional state. Forward losses/logits agreed, while the backward graph boundaries differed.

The corrected path constructs positional state from a detached hidden boundary. Attention masks are value-independent in the tested path. No evidence supports claiming a genuine semantic positional derivative; this was a reference graph mismatch. The T conventional reference retains the original hidden leaf and compares its gradient against the manual external-slot VJP.
