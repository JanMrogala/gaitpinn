# Plan, study 2: one model for many people

Written before any study-2 model was trained, and revised once after an independent review,
still before any training. Changes made after results have been seen are logged at the bottom
with the reason. Study 1 (`PLAN.md`) and its results are not changed by this study.

## Question

One model is trained on many people and used on people it has never seen, without any labelled
data from them. Compared on those new people:

| Name | What it is |
|---|---|
| `base` | the network trained on many people, angle error only |
| `pinn` | the same network trained with angle error plus the physics loss |
| `pinn_cal` | `pinn`, then self-calibrated to the new person: fine-tuned on 2 minutes of that person's **unlabelled** walking with physics losses and an anchor (no true angles) |
| `base_cal` (control) | `base`, self-calibrated the same way with the gyro term only, which needs no torque output. It shows whether label-free adaptation helps without physics training. |

The questions:

1. Does physics during training help on new people (`base` vs `pinn`)?
2. Is the self-calibrated physics model better than the plain data model (`base` vs `pinn_cal`)?
3. Does the self-calibration itself help (`pinn` vs `pinn_cal`)?

The test is study 1's stream: walking, then the person's stair recordings. Training people
contribute walking, ramps **and** stairs, so stairs are not a new task here. Only the person is
new.

**Main result:** error over time since the switch to stairs, with `base`, `pinn` and
`pinn_cal`, for people the models never saw.

## Data

- **Dataset and processing:** Camargo et al. 2021, processed as in study 1 (causal filters,
  100 Hz, same 35 input channels, same target-quality rule). Ramp files are converted too,
  into a separate folder (`data/h5x`); study 1's data are untouched.
  - Note: the target-quality rule reads targets to drop broken files, for test people too.
    It only removes files and never feeds a model.
- **People:** all 22. Nobody is excluded as a test person.
  - The six people who failed study 1's gyro check (AB06, AB07, AB08, AB16, AB19, AB20) add
    only the angle loss, no physics terms, when they are training people.
- **Per-person walking split:** the same as in study 1 (by file, seed 0).

### Folds (fixed now, from `cross.folds`, seed 0)

| Fold | Test people | Validation people | Training people |
|---|---|---|---|
| 0 | AB08, AB10, AB16, AB18, AB23, AB27 | AB07, AB15, AB24 | the other 13 |
| 1 | AB06, AB09, AB12, AB14, AB17, AB28 | AB16, AB18, AB21 | the other 13 |
| 2 | AB11, AB13, AB19, AB25, AB30 | AB09, AB18, AB23 | the other 14 |
| 3 | AB07, AB15, AB20, AB21, AB24 | AB06, AB08, AB16 | the other 14 |

Every person is a test person exactly once. **All choices are nested within a fold.** A fold's
configuration and calibration settings are chosen from that fold's validation people only, so
a test person's labels never feed a choice that is applied to them.

### One label-free recipe for every person

The following come from sensor signals only, the same way for training, validation and test
people:

- **Input normalization:** per-channel mean and std from the person's walking train files.
- **Calibration walk:** 120 s of the person's first treadmill walking train file, starting 10 s
  in (steady walking). People with no treadmill train file use level-ground files instead.
- **From the calibration walk:**
  - EMG scaling for the activations (95th percentile);
  - swing threshold (foot gyro);
  - gyro scales, as the std of the person's smoothed gyro joint speed.
- **Swing scales:**
  - `tau_scale` = m·g·d from height and mass;
  - `scale_dyn` = `tau_scale` × R, where R is the median of (true-angle swing scale ÷
    `tau_scale`) over the fold's training people.
- **Target normalization:** one global mean and std from the fold's training people.

**Gyro alignment:**

- **Training people** keep study 1's label-fitted alignment (labels allowed for them).
- **Validation and test people**, when calibrated, get a label-free alignment:
  - the principal axis of each sensor's gyro on the calibration walk;
  - one sign per sensor (thigh, shank, foot), chosen jointly for knee and ankle by
    correlation with the original model's predicted joint speeds.
- **Gate:** a joint's gyro term is used only if that correlation is ≥ 0.8. On validation
  people, how often the gate agrees with the true-angle check is reported.

## Models

- **Network:** the same causal 1D CNN as study 1.
- **Settings:** identical for all models.
  - Dropout 0, batch 64, 4 s chunks.
  - 100 steps per epoch, at most 100 epochs, patience 8.
- **Training chunks:** a person is drawn in proportion to their data, then a file, then a chunk.
- **Selection score:** the mean over validation people of each person's angle RMSE over all
  their files, averaged over hip, knee and ankle. Used for early stopping and for choices.

### `pinn` physics loss (training)

- **Terms:** the same gyro and swing terms as study 1, with each chunk's person's parameters.
- **Shared parameters:** learned and scaled by each person's body size: muscle torque =
  `tau_scale`·Σ wₘaₘ, stiffness = `tau_scale`·k, damping = `tau_scale`·b.
- **λ grid:** {0.001, 0.01, 0.1}, a wider span than study 1. Across many people the angle loss
  is larger, so the same λ is relatively weaker. The grid now reaches the strong value that
  study 1 tested separately.

### Self-calibration (`pinn_cal`, `base_cal`; validation and test people only)

- **Starting point:** a copy of the trained model, fine-tuned on the calibration walk with
  Adam (lr 1e-4), batch 64, 4 s chunks, seeded with the model's seed.
- **Updated:** network weights only; the physics parameters stay frozen.
- **Loss:**
  `Σ_terms L_t / L_t(start) + μ · anchor / anchor_ref`
  - Each physics term is divided by its value at the start, so no term dominates by scale.
  - The anchor is the mean squared difference between the calibrated and the frozen original
    model's predicted angles, in normalized units.
  - `anchor_ref` is the anchor for a 1° difference.
- **Settings per fold**, chosen on that fold's validation people through the same label-free
  path:
  - `pinn_cal`: terms {gyro, gyro + swing} × steps {100, 300} × μ {1, 10}, plus "no
    calibration".
  - `base_cal`: gyro × steps {100, 300} × μ {1, 10}, plus "no calibration".
- **If "no calibration" wins** in a fold, that fold's calibrated model is simply the
  uncalibrated one. This is reported.
- **Diagnostic on validation people:** each physics term is evaluated at the true angles and at
  the original predictions. If the true angles do not score lower, calibration cannot be
  expected to help, and a null or negative `pinn_cal` result is the expected outcome.

### Tuning (same effort for `base` and `pinn`)

- **Configurations:** 12 each.
  - Shared: lr {1e-3, 3e-3} × width {64, 128}.
  - `base` adds weight decay {0, 1e-5, 1e-4}.
  - `pinn` adds λ {0.001, 0.01, 0.1}.
- **Runs:** seed 100, every fold.
- **Choice:** per fold, the lowest selection score on that fold's validation people.
- **Final runs:** per fold, 5 seeds (0–4) of `base` and `pinn` with the fold's choice. Every
  final `pinn` (and `base`) run is calibrated for each of the fold's test people with the
  fold's chosen settings.

## Metrics and test

Per test person and model, averaged over 5 seeds, with study 1's definitions:

- error after switch (RMSE over the stairs part);
- rolling 2 s error;
- walking error `W` (level-ground walking test files);
- climbing-up and climbing-down errors;
- time to fail (secondary only).

**Primary endpoint:** error after switch, averaged over hip, knee and ankle.

**Primary tests:** a family of three, paired across the 22 people, two-sided Wilcoxon
signed-rank with `zero_method="wilcox"`, Holm-corrected:

1. `base − pinn`
2. `base − pinn_cal`
3. `pinn − pinn_cal`

- **Effect sizes:** rank-biserial r, and the Hodges–Lehmann difference with a 95% bootstrap CI.
- **Verdict per test:** a model is "better" if its Holm-adjusted p < 0.05 with r in its
  favour; otherwise "no clear difference".
- **Limitation:** the people in a fold share the same trained models, so the 22 pairs are not
  fully independent. Per-fold median differences are reported next to the pooled test.

**Secondary results:**

- the `base_cal` control (`base` vs `base_cal`, `pinn_cal` vs `base_cal`);
- per joint, climbing up and down, time to fail;
- the 16 people who passed the gyro check;
- how often each fold chose "no calibration", and how often the gate dropped a joint;
- the calibration diagnostic.

**Plots:** study 1's plots, with `base`, `pinn` and `pinn_cal`.

## Compute

- **Partition:** `small-g`, 1 GCD per task.
- **Tasks:**
  - tuning: one array task per (model, fold, configuration), 96 tasks;
  - calibration choice: one per fold, 4 tasks;
  - final: one per (model, fold, seed), 40 tasks.
- **GPU-hours:** estimated after timing the first tuning tasks. Expected well under 50,
  against about 950 left; the ledger keeps the running total.

## Decisions log

- **2026-11-19, study 2 defined** with the user: option B (training people contribute all
  tasks), three models, held-out people. Ramps converted into `data/h5x`.
- **2026-11-19, plan revised after an independent review, before any training.** Changes:
  - all choices nested within folds;
  - one label-free recipe for every person;
  - normalized calibration losses, a gyro-only option, a "no calibration" option and a
    label-free gate;
  - the `pinn − pinn_cal` test and the `base_cal` control added;
  - the λ grid widened;
  - body-size-scaled shared swing parameters;
  - the fold table fixed in the plan.
- **2026-11-19, local development runs.** Short runs on 4 people (AB21, AB23, AB24, AB25),
  copied locally only to test the code, produced stream predictions for AB24, a fold-0 test
  person. Nobody looked at them, and they were deleted. The code now refuses to run if the folds
  differ from the table above, and refuses to hand on incomplete per-fold choices.
- **2026-11-19, pilot (tuning tasks 3 and 16, fold 0, array 23869610).**
  - Timing: about 4 min per run (prep 18 s, training about 200 s). Early stopping hit after
    17–18 epochs for both models, far below the cap.
  - The fold's 10 physics people match study 1's gyro check.
  - At λ = 0.01 the physics part is about half the angle loss, so the grid spans weak to strong.
  - Estimate for the remaining 94 tuning tasks: about 4 GPU-hours, at most 47 with the 1 h
    limit.
  - An earlier, mistaken full-array submission (23869604) was cancelled before any task
    started.
  - The pilot runs are genuine tuning runs at the same commit; the full batch skips their
    indices.
- **2026-11-19, tuning done (96 runs, arrays 23869610 and 23869690; one task held by a LUMI
  filesystem error and released unchanged).** Per-fold choices (from `runs/chosen_x.json`,
  lowest validation score):

  | Fold | `base` | `pinn` |
  |---|---|---|
  | 0 | lr 1e-3, width 128, wd 1e-4 | lr 1e-3, width 128, λ 0.001 |
  | 1 | lr 3e-3, width 128, wd 1e-5 | lr 1e-3, width 128, λ 0.001 |
  | 2 | lr 3e-3, width 64, wd 0 | lr 1e-3, width 128, λ 0.001 |
  | 3 | lr 3e-3, width 128, wd 1e-4 | lr 3e-3, width 128, λ 0.001 |

  - In every fold, `pinn`'s validation score gets worse as λ grows, and the smallest λ in
    the grid wins. The grid stays as planned; it is not extended, so both models keep the
    same tuning effort. The physics loss in the final `pinn` is therefore weak.
  - Code fixes with no effect on any number: the `__main__` block in `test.py` moved below
    the study-2 functions, and early smoke records with no `status` field are skipped.
