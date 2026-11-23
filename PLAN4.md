# Plan, study 4 (DRAFT): gyros only in the physics loss

Status: **draft.** Not reviewed and not frozen. A feasibility pilot (below) runs first. Choices
the user left to the recommendation are marked **[decided]**. Studies 1–3 are not changed by
this study.

## Why this study

In studies 1–3, the physics loss added little that the network could not already get:

- **The gyro term** asks that d(angle)/dt match the gyro difference across the joint. The
  gyros were also network inputs, so the network could learn that relation from a few labelled
  seconds, and the term added no new information.
- **The swing term** uses a pendulum model that explains real knee swing poorly (study 1: even
  the true angles leave a residual of about 2.6× the signal).

A physics loss can only help when it carries information the inputs and the remaining labels
do not.

**Idea:** the network does not see the gyros. They are used only inside the physics loss,
during training. This matches a real use:
- train in a lab where IMUs, and sometimes motion capture, are available;
- deploy a cheaper device without gyros.

## Question

With the gyros removed from the network's inputs, and with a share of the training angles
missing (study 3's gaps), does a network trained with the gyro physics term keep a lower stair
error on new people than the same network trained on the remaining angles only?

The gyro term only adds information where true angles are missing: where angles exist, they
already contain d(angle)/dt. So the study keeps study 3's missing-angle shares. The effect is
expected to grow with the share.

## Models

| Name | Network inputs | Training |
|---|---|---|
| `base` | inputs without gyros (below) | remaining true angles only |
| `pinn` | same | remaining angles + gyro term on all data (labelled and unlabelled batches) |
| `distill` (secondary) **[decide]** | same | remaining angles + pseudo-angles from a teacher. The teacher is study 3's all-sensor `base` at the same share, applied to the stretches without angles. It is a non-physics way to use the gyros, and the fair competitor for `pinn`. |
| reference (not tested) | all 35 channels | study 3's `base` and `pinn` at the same share: what the gyros give as inputs |

**Network inputs [decide], one primary set:**
- **A, EMG + accelerometers (23 channels):** an IMU without gyro. Accelerometers still show
  segment tilt, so the gyro adds partly redundant information.
- **B, EMG only (11 channels):** the largest gap between what the network sees and what the
  physics knows. Errors will be much larger for both models.
- Recommendation: B as primary (the cleanest test of "physics carries privileged
  information"), A as secondary.

**Physics term:**
- Only the gyro term for knee and ankle, with study 3's calibration-trial alignment. The swing
  term is dropped.
- **[decide]** Add a hip term: thigh gyro − trunk gyro, as an approximation. There is no pelvis
  IMU, and the trunk differs from the pelvis by the lumbar angle.

## Setup (from study 3, unchanged unless stated)

- Same data, people, folds and test stream.
- Missing shares 0%, 50%, 90%, 99%, with the same masks and draws, and the 30 s calibration
  trial.
- Same sampler, separate random streams, validation score, no-leak test, tuning grids
  (16 configurations each), smallest effect of interest (0.5°), tests (Holm over 50%, 90%,
  99%) and verdict rules.
- **Input normalization:** over the chosen input channels only.
- **Test people's gyros:** never used, not even for normalization.

## Primary result

Error on stairs against the share of missing angles, with `base` and `pinn` (input set B),
plus study 3's all-sensor models as a reference line.

## Compute (estimate)

- **Tuning:** 4 shares × 4 folds × 2 models × 16 configurations = 512 runs.
- **Finals:** 160 runs, plus `distill` 60 and set A 160 if chosen.
- **Estimate:** about 40–60 GPU-hours, similar to study 3.

## Open decisions

1. Primary input set: B (EMG only) or A (EMG + accelerometers).
2. Include `distill`.
3. Add the approximate hip term.
4. Keep all four shares or drop 50% to save compute.

## Feasibility pilot (before the plan is frozen)

On 2026-11-23 the user asked to run study 4 with a minimal training run first, to check that
the study makes sense. The four open decisions were left to the recommendations:

- **[decided]** EMG-only inputs as primary, EMG + accelerometers as secondary.
- **[decided]** No hip term.
- **[decided]** All four shares kept.
- **[decided]** `distill` decided after the pilot.

The pilot (20 runs, `gaps.PILOT4`, kind `g4pilot`):
- **Scope:** fold 0 only, seed and mask draw 100, scored on fold 0's validation people.
- **Network:** lr 3e-3, width 128, no swing term.
- **EMG only,** at 0%, 90% and 99% missing:
  - `base` with weight decay 0 and 1e-4;
  - `pinn` with λ 0.01, 0.1, 1 and 10.
- **EMG + accelerometers,** at 99% missing: `base`, and `pinn` with λ 1.

**What would make the study worth running:**
1. At 90–99% missing, EMG-only `base` is clearly worse than study 3's all-sensor `base` (fold 0
   tuning runs with the same lr and width). Otherwise the gyros carry nothing the network
   lacks.
2. At 90–99%, some λ gives `pinn` a lower validation score than `base`. Otherwise the gyro
   term does not transfer that information.

**If both hold:**
- the λ grid for the full study is set from the pilot;
- the plan is frozen after an independent review;
- fold 0's validation people are used again for fold 0's choices (declared here; no test
  people are involved).

**If either fails:** study 4 stops, and the pilot is reported as the result.
