# LITE-008Q bridge architecture decision

Primary: shared Float32 `Linear(384,576)` bridge, 221,760 parameters, consuming frozen L2-normalized MiniLM evidence features and producing one decoder-compatible slot. Initialization is decoder-scale: small weight noise (`sigma=0.0005`) and the frozen EOS embedding as bias.

Comparative control: one shared 576-dimensional memory-independent learned soft prompt. It receives no factual evidence. No entity-specific lookup table, nonlinear parallel bridge, MiniLM update, or decoder update was implemented.
