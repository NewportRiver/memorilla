# LITE-008U student architecture decision

Selected baseline: frozen MiniLM 384-D evidence feature -> residual linear W[576,384] -> frozen EOS reference -> one 576-D latent slot. Trainable parameter count: 221,184. EOS and all decoder/encoder parameters were frozen.

No query-conditioned or nonlinear comparative architecture was authorized because the primary student failed the mandatory training semantic gate and the 256-update allowance was exhausted.
