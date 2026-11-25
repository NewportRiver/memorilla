# LITE-008R prior implementation audit

Baseline: `83b6f34`. Working tree was clean and branch was `feature/memorilla-lite-android`. All LITE-008Q reports and trajectories remain preserved.

The Q bridge implementation is `experiments/lite008q/bridge_train.py:30-46`. It constructs `Linear(384,576)`, initializes bias to EOS, computes a manual decoder objective gradient, calls `slot.backward(pre["grad"])`, clips bridge parameters, and updates Adam. Q did not numerically qualify the manual VJP before training.

The historical gradient harness is `experiments/lite008q/gradient_chain.py:9-18`. It perturbs flattened parameter views, restores values, but compares finite differences of `layerwise(...)["loss"]` with a gradient obtained from the manual external-slot VJP. Q retained five coordinates `[0,384,1000,10000,200000]`, epsilon `0.01`, and maximum relative error `1.000549`.

Observed Q scale facts: output norm min/max `20.259939/103.399994`, bridge gradient norm min/max `0.159473/23.209023`, sampled RSS max `629215232`, and minimum available memory `1280737280`.

R changes add `experiments/lite008r/gradient_qualification.py`, a fail-closed `require_qualification()` in the Q training wrapper, and regression tests. No model weights, caches, production state or authoritative evidence were changed.
