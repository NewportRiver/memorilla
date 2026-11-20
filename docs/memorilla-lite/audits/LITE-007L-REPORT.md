# LITE-007L Corrected Comparative Training Report

## Authorization and scope

LITE-007L was explicitly authorized by the owner from baseline commit `da0c3b3` on `feature/memorilla-lite-android`. The experiment evaluates query-conditioned evidence ranking, not language-model latent conditioning. No holdout, external inference, APEX data, production integration, brokerage activity, or LITE-008 implementation was used.

## Execution

- Associative updates: 240/240
- Direct updates: 240/240
- Combined updates: 480/480
- Episodes: 30 per model
- Updates per episode: 8
- Optimizer: FP32 SGD, learning rate 0.001, clip norm 1.0, batch size 1
- Training exposure: 240 unique eligible examples per model; 80 each at collection sizes 8, 16, and 32
- Scheduler: persistent, identity-bound, matched order digest `e662983eaf11c3f86b1b425c56d426fc6fecd8dd7b6c226903b328a4655bcd73`
- Initialization: fresh associative; explicit zero direct matrix

The first combined runner failed before direct optimizer work because it passed no direct initialization checkpoint. That failed ledger is preserved. A fresh explicit zero checkpoint was then used with a separate recovered authorization ledger. The recovered direct run completed exactly 240 updates; the failed launch consumed zero updates and zero scheduler reservations.

## Evaluation

Milestones were evaluated at 0, 40, 80, 120, 160, 200, and 240 updates on two separate development partitions:

1. Independent corpus: collection sizes 8, 16, 32; no expected-abstention examples, so abstention evaluation is unavailable.
2. Existing larger development-validation corpus: collection sizes 64, 128.

Final independent answerable results at 240 updates:

| Model | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | MRR | Complete@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Associative | 0.1222 | 0.2250 | 0.3167 | 0.0722 | 0.1472 | 0.2250 | 0.2446 | 0.1333 |
| Direct | 0.8861 | 0.9778 | 1.0000 | 0.5542 | 0.6181 | 0.6681 | 0.9351 | 0.3361 |
| Cosine | See comparative JSON | See comparative JSON | See comparative JSON | See comparative JSON | See comparative JSON | See comparative JSON | See comparative JSON | See comparative JSON |

The direct model is materially stronger on the independent development corpus. The associative model does not meet the declared superiority thresholds.

On the larger development-validation corpus, the associative model reaches Complete@5 of 1.0 at both 64 and 128 records, but this is a development result and is not combined with the independent-corpus claim. The full category and collection-size breakdowns are in `LITE-007L-COMPARATIVE-EVALUATION.json`.

## Ordering controls

The final associative model was invariant under the evaluated evidence-order reversal. The direct final model also produced identical aggregate metrics under reversal, with six tie rows. The zero direct initial model was strongly order-sensitive: ordinary ordering produced Hit@1 1.0, while reversed ordering produced Hit@1 0.0. This confirms the declared tie-breaking risk and prevents interpreting the zero control as initialization-independent evidence.

The six normalized-template collision groups and their scenario/partition relationships are recorded in `LITE-007L-ORDERING-CONTROLS.json`.

## Representation analysis

The associative representation audit reports slot similarity/diversity, attention entropy, required-evidence attention relative to uniform, query sensitivity, and evidence-order sensitivity. These measurements are reported separately from ranking results. No decoder-conditioning or latent-language-model claim is made.

## Engineering disposition

Training exposure, authorization accounting, persistent scheduling, checkpoint persistence, matched order, milestone evaluation, and read-only ordering controls completed. Resource telemetry is recorded per episode in `LITE-007L-RESOURCE-TELEMETRY.json`. The complete regression result and final checkpoint hashes are recorded in the corresponding audit artifacts.

## Scientific disposition

Negative result for the primary associative-superiority objective. Broad deterministic exposure was achieved, but the associative model did not outperform the trainable direct ranking control on the independent development corpus. The larger-corpus Complete@5 ceiling remains a potential dataset shortcut and is not evidence of general-world generalization.

Recommendation: keep LITE-008 blocked. Preserve the negative/inconclusive result and require owner review before any further architecture or decoder-conditioning work.
