# LITE-008G Projection Mathematics Audit

## Finding

LITE-008F reported `L = mean((prediction-target)^2)` but `experiments/lite008f/projection.py` implemented:

- `grad_w = X^T error / batch_size`
- `grad_b = mean(error)`

For an output width of 576, the correct gradients of the reported elementwise MSE are:

- `grad_w = 2 X^T error / (batch_size * output_dim)`
- `grad_b = 2 sum(error) / (batch_size * output_dim)`

Therefore the historical update was `output_dim/2 = 288` times larger than the correctly normalized MSE gradient for the same nominal learning rate. The historical `lr=0.08` therefore has a mathematically equivalent corrected-MSE learning rate of `23.04`.

## Numerical verification

The corrected `mse_gradients()` was checked against central finite differences with `h=1e-3`, covering both weight and bias parameters at batch sizes 1, 2, and 7 and output widths 3, 8, and 576. Maximum absolute errors were:

- batch 1: weight `5.80e-5`, bias `5.32e-5`
- batch 2: weight `8.44e-5`, bias `7.06e-5`
- batch 7: weight `8.54e-5`, bias `3.98e-5`

The differences are consistent with Float32 perturbation and large output reduction. The corrected path passed.

The named `train_sgd_legacy_unscaled()` method preserves the original LITE-008F update rule. A 10-step trajectory using historical `lr=0.05` was compared with corrected MSE training at `lr=0.05 * output_dim / 2`; maximum parameter differences were `7.45e-9` for both weights and bias.

## Historical preservation

LITE-008F results are not rewritten. The LITE-008G corrected projection uses the mathematically correct gradients and equivalent scaling for controlled comparison. Historical and corrected outputs are reported separately in the semantic results.
