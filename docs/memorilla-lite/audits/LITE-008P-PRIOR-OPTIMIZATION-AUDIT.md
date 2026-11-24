# LITE-008P Prior Optimization Audit

The O implementation correctly froze decoder parameters, injected one slot at causal position zero, used Float32 master slots, computed complete answer-token likelihoods, and manually propagated the external gradient through all 30 layers. Its update records show Adam was instantiated per isolated embedding, gradient clipping was applied at norm 1.0, and the O runs used learning rate 0.05.

The P audit found that O's update-eight stopping point was premature for the observed trajectories. O A directions often improved without crossing zero, while B directions were sometimes still moving at update eight. P therefore extended calibration to update 16 before selecting a fresh-oracle configuration. The P harness initially exposed and corrected a batch-dimension gradient extraction defect before any P optimizer update completed; that failed invocation is preserved as an implementation event, not scientific evidence.
