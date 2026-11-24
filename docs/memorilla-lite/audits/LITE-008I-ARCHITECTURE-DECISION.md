# LITE-008I Architecture Decision

Preserve the original differentiable architecture as the scientific reference:

`document/question embeddings -> MemoryModule -> injected decoder input embeddings -> frozen causal decoder -> answer-only loss -> gradients into MemoryModule`

Do not modify the qualified Q4_K_M inference path or production MemoryModule artifacts.

The tiny diagnostic confirms the critical autograd mechanism, but full SmolLM2 differentiation is blocked by the current ceiling. The correct next architectural step is not a new optimizer or reduced scientific objective; it is owner review of a narrowly scoped training-only memory envelope or an explicitly approved reduced decoder reference. No general bridge training begins here.
