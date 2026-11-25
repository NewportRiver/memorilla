# LITE-008T bridge parameterization

Primary architecture retained: one shared linear projection from MiniLM dimension 384 to decoder dimension 576. The selected calibration parameterization was `z = EOS_embedding + W*x`, with frozen EOS embedding and zero-initialized residual W. MiniLM feature norms were measured at approximately 1.0 because the configured encoder L2-normalizes outputs.

The residual initialization produced EOS-scale latent norms. The 0.0001 Adam arm remained closer to EOS scale than 0.001. No larger network or extra slot was introduced.
