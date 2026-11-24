# LITE-008N Frozen Decoder Memory Audit

The prior complete FP32 load exceeded the 1 GiB RSS ceiling because the complete model was simultaneously resident in Float32. LITE-008N measured a different boundary: official SmolLM2-135M safetensors remain in BF16 storage, while only the active frozen decoder layer is copied to Float32 for computation.

Measured phase RSS:

- After BF16 model load: 346,533,888 bytes
- After Float32 input embedding: 347,222,016 bytes
- After 30-layer forward: 557,424,640 bytes
- Forward peak: 571,572,224 bytes
- After backward VJP: 700,932,096 bytes
- Backward peak: 743,833,600 bytes
- Monitored peak RSS: 741,912,576 bytes
- Peak PSS: 733,217,792 bytes
- Minimum available memory: 1,094,270,976 bytes

The principal savings are avoiding simultaneous Float32 parameter residency and avoiding autograd retention of decoder parameter gradients. The layer-wise method stores detached hidden boundaries, reconstructs the active layer during reverse VJP, and releases the temporary Float32 layer before proceeding.

The existing FP32 full-load failure and all LITE-008M reports remain unchanged.
