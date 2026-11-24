# Plan, study 4: gyros only in the physics loss

Written before any study-4 tuning or final run, after a small feasibility pilot (below).
Changes made after results have been seen are logged at the bottom with the reason. Studies 1–3
are not changed by this study.

## Why this study

In studies 1–3, the physics loss added nothing the network could not already get:

- **The gyro term** asks that d(angle)/dt match the gyro difference across the joint. The
  gyros were also network inputs, so the network could learn that relation from a few labelled
  seconds.
- **The swing term** uses a pendulum model that explains real knee swing poorly.
- **Study 3** found `base` better than `pinn` at every share of missing angles.

A physics loss can only help when it carries information that neither the inputs nor the
remaining labels contain.

**Idea:** the network does not see the gyros. They are used only inside the physics loss,
during training. This matches a real use:
- train in a lab where IMUs, and sometimes motion capture, are available;
- deploy a cheaper device without gyros.

The gyro term can only add information where true angles are missing, so the study keeps
study 3's missing-angle shares.

## Question

With the gyros removed from the network's inputs, does a network trained with the gyro physics
term keep a lower stair error on new people than the same network trained on the remaining
angles only, as more of the training angles are missing?

## Models

| Name | Network inputs | Training |
|---|---|---|
| `base` | no gyros (input set below) | remaining true angles only |
| `pinn` | same | remaining angles + gyro term on all data (labelled and unlabelled batches); no swing term |
| `distill` (secondary) | same | remaining angles + a teacher's predictions in place of the lost angles. The teacher is study 3's all-sensor `base` configuration for that share and fold, retrained in the same task with the same seed and mask. It is a non-physics way to use the extra sensors, but not a perfectly matched competitor (see below). Not tuned: it uses `base`'s choice. |
| reference (not tested) | all 35 channels | study 3's `base` at the same share |

**Input sets:** two, each run as a full study with its own tuning, choices and tests.
- **Primary, EMG only (11 channels), prefix `g4`.**
- **Secondary, EMG + accelerometers (23 channels), prefix `g4a`.** An IMU without gyro;
  accelerometers still show segment tilt.

**Physics term:** the gyro term for knee and ankle only, with study 3's calibration-trial
alignment and physics people. No hip term, no swing term, no torque output.

## Setup (from study 3, unchanged unless stated)

- Same data, people, folds and test stream.
- Shares 0%, 50%, 90% and 99%, with the same masks, draws and 30 s calibration trial.
- Same sampler, separate random streams, validation score, no-leak tests (now also covering
  `distill`), smallest effect of interest (0.5°), tests and verdict rules.
- **Inputs:** the network takes the chosen channels only. Normalization statistics are per
  channel, so the unused channels' statistics are computed but never reach the network.
  - A test checks that the network's output does not change when the gyro channels change.
- **`distill` targets:** where an angle is kept, the true angle; where it is lost, the
  teacher's prediction. Chunks are drawn from all samples, and the target normalization uses
  the kept angles only.

## Tuning (same effort for `base` and `pinn`)

- Per input set, share and fold: 16 configurations per model at seed 100.
  - Shared: lr {1e-3, 3e-3} × width {64, 128}.
  - `base` adds weight decay {0, 1e-5, 1e-4, 1e-3}.
  - `pinn` adds λ {0.01, 0.1, 0.3, 1}.
- **λ grid from the pilot:** λ = 10 was worse everywhere, and 0.01–1 covered the best values.
- **Choice:** the lowest validation score per share and fold.

## Final runs, metrics and tests

**Headroom.** Losing angles can only cost so much: in study 3, the all-sensor `base`'s stair
error rose by only 0.07° at 50%, 0.15° at 90% and 0.74° at 99%.
- The **headroom** at a share is `base`@share − `base`@0%: the EMG-only `base` stair error,
  mean of hip, knee and ankle, averaged over people, with a 95% bootstrap CI over people.
- `pinn`'s gain is reported next to the headroom.
- **"Equivalent"** is interpreted only where the headroom is at least 0.5°.
- **A gain larger than the headroom** (`pinn` beating the full-label `base`) counts as
  regularization, not as recovering lost information.


- **Final runs:** per input set, share, fold and model, the chosen configuration with seeds
  0–4 (= mask draws). `distill`: EMG-only set, shares 50%, 90% and 99%.
- **Primary endpoint:** error after switch to stairs, mean of hip, knee and ankle.
- **Primary test (single):** EMG-only set, 99% missing, `base − pinn`, paired over the 22
  people.
  - Two-sided Wilcoxon signed-rank (`zero_method="wilcox"`), α = 0.05.
  - 99% is the only share with real headroom.
  - Verdicts as in study 3: "better" needs p < 0.05 and the per-fold median favouring the
    same model in at least 3 of 4 folds; "equivalent" means the Hodges–Lehmann CI lies within
    ±0.5°; otherwise "inconclusive".
- If λ = 0.01 (the smallest) wins at a share, the result is reported as "validation
  preferred almost no physics".
- **Secondary:**
  - 50% and 90% (EMG only), Holm-corrected as a family of two;
  - the same tests for the EMG + accelerometers set;
  - **`pinn` vs `distill`** (EMG only): Wilcoxon per share at 50%, 90% and 99%,
    Holm-corrected as its own family, with study 3's verdicts.
    - Reading: `distill` ≥ `pinn` means the physics is not needed to use the extra
      sensors.
    - `pinn` > `distill` only means `pinn` beats an untuned teacher–student whose teacher
      sees more. The teacher sees all 35 channels (accelerometers too, for the EMG-only
      set) and every training person's gyros, while `pinn` uses only the physics people's
      gyros, and only through the knee and ankle terms.
  - the teacher's validation score against study 3's `base` final run for the same share,
    fold and seed (they should agree up to GPU noise);
  - per-joint stair error, especially knee and ankle: the gyro term covers only those two
    joints, while the primary endpoint also includes the hip;
  - the 0% share;
  - walking error, per joint, climbing up and down;
  - the 16 people who passed study 1's gyro check;
  - the Page trend test over 50%, 90% and 99% (does the `base − pinn` difference grow with
    the share?).

## Disclosures

- The pilot used fold 0's validation people at the tuning seed 100, with configurations that
  are in the full grid. `pinn`'s λ grid (not `base`'s) was adapted from it. The 3-of-4
  fold-agreement rule limits what this can do.
- The 22 test people are the same as in studies 1–3, whose results motivated this study.
- No placebo control: a gyro reference shifted in time would rule out "any derivative penalty
  would do". It was considered and left out.

## Compute

- **Tuning:** 2 input sets × 512 = 1024 runs.
- **Finals:** 2 × 160 + 60 `distill` = 380 runs.
- **Estimate:**
  - EMG-only runs took 1–4 min of training in the pilot, so about 35 GPU-hours for tuning
    and about 25 for the finals.
  - Ledger before study 4: 40.8 GPU-hours for the whole project.
- **Queue:** submitted in chunks of at most 200 tasks (LUMI's limit is about 210 per user).

## Feasibility pilot (2026-11-23, before this plan was frozen)

The user asked to run study 4 with a minimal training run first, to check that it makes sense.
The open decisions were left to the recommendations:
- EMG only as primary, EMG + accelerometers as secondary;
- no hip term;
- all four shares;
- `distill` decided after the pilot, and now included.

**Pilot design:** 20 runs (`gaps.PILOT4`, kind `g4pilot`, array 23968063).
- fold 0, seed and draw 100, lr 3e-3, width 128;
- scored on fold 0's validation people only; no test people involved.

**Conditions written before the pilot, and results** (validation RMSE, °):

| | 0% | 90% | 99% |
|---|---|---|---|
| all-sensor `base` (study 3 tuning, same lr and width) | 6.24 | 6.63 | 6.13 |
| EMG-only `base`, wd 0 / 1e-4 | 8.06 / 7.99 | 8.29 / 8.31 | 8.32 / 8.31 |
| EMG-only `pinn`, λ 0.01 / 0.1 / 1 / 10 | 8.05 / 8.10 / 8.10 / 8.89 | 8.20 / 8.21 / 8.55 / 9.38 | 8.08 / 7.97 / 7.97 / 8.31 |
| EMG + accelerometers at 99%: `base` / `pinn` (λ 1) | | | 6.46 / 6.17 |

1. **EMG-only `base` is clearly worse than the all-sensor `base`** (by about 1.7–2.2°), so
   the gyros carry information the EMG-only network lacks. **Met.**
2. **Some λ gives `pinn` a lower validation score than `base` at 90–99%:**
   - 90%: by 0.09°;
   - 99%: by 0.34°, with two λ values agreeing;
   - EMG + accelerometers at 99%: by 0.29°.
   **Met.**

**Caveats:**
- The pilot is one fold and one seed, and GPU run-to-run noise reached 0.5° in study 2.
- It tried four λ values against two weight decays, which favours `pinn` slightly.
- Only the full study can confirm the effect.

Fold 0's validation people are used again for fold 0's choices in the full study; no test
people were involved in the pilot.

## Decisions log

- **2026-11-23, study 4 drafted** after study 3's tuning showed validation preferring almost
  no physics.
- **2026-11-24, feasibility pilot passed** (above).
  - Plan completed:
    - λ grid {0.01, 0.1, 0.3, 1};
    - both input sets with full tuning;
    - `distill` included.
  - Code:
    - input-channel choice and swing switch (`cross.XModel`);
    - per-study prefixes for configs, record kinds and choices;
    - `distill` through teacher fill-ins (`GapPerson.fill`);
    - tests that the gyros never reach the network and that lost angles never change
      `pinn` or `distill` training.
- **2026-11-24, plan revised after an independent review, before any study-4 tuning run:**
  - 99% is the single primary share (study 3 showed almost no headroom at 50% and 90%);
  - headroom defined and reported;
  - smallest-λ rule;
  - `pinn` vs `distill` test specified, and `distill` described accurately;
  - per-joint secondary;
  - disclosures.
  - Code: `gdistill` refuses to run as study 3 (`--prefix g`); job names carry the study
    prefix; the lost-angle test also runs the study-4 `pinn` setup. The study-4 analysis
    (`gsummary` with a prefix, `distill` and headroom) is committed before any final run
    is scored.
- **2026-11-24, tuning done (1024 runs, all completed: arrays 24003443, 24005578, 24006043,
  24006413, 24006948 and 24007657). Choices frozen in `runs/chosen_g4.json` and
  `runs/chosen_g4a.json` before any final run.**
  - The first array ran at commit 822c33c and the others at 4929d87. The two commits differ
    only in the analysis, plots and website, not in training.
  - **λ:**
    - EMG only: 0.01 (the smallest, "validation preferred almost no physics") won at 0%
      folds 0 and 2. Elsewhere 0.1–1 won, and 1 (the top value) won at 0% fold 1, 50% fold 3
      and 90% fold 1.
    - EMG + accelerometers: 0.01 won at 90% fold 2 and 99% folds 2 and 3. 1 won at 50% fold 1.
  - **Best validation score, `base` − `pinn` (°, folds 0–3):**

    | Missing | EMG only | EMG + accelerometers |
    |---|---|---|
    | 0% | −0.02, +0.07, +0.04, +0.22 | +0.07, −0.03, +0.05, +0.36 |
    | 50% | +0.18, +0.17, +0.05, +0.26 | +0.13, +0.22, −0.17, +0.09 |
    | 90% | −0.03, +0.10, −0.10, +0.18 | −0.00, +0.21, −0.03, −0.03 |
    | 99% | +0.48, +0.03, +0.51, +0.25 | +0.41, +0.24, +0.13, +0.35 |

    These are the best of 16 configurations per model at one seed, on validation people. They
    are not the test.
  - **GPU-hours:** 27.9 for tuning; ledger 68.7 for the whole project.
  - **Finals submitted with lower time limits** (30 min per final run, 1 h per `distill` run
    with its teacher; the longest tuning run took 7.5 min). Worst case about 110 GPU-hours.
- **2026-11-25, final runs done** (arrays 24008564, 24009038 and 24009276, all 380 tasks
  completed; ledger 83.2 GPU-hours for the whole project).
  - **Result:** at 99% missing, EMG only, `pinn` is better (Hodges–Lehmann +0.34°, CI +0.15
    to +0.58, p = 0.003, all four folds), below the 0.5° smallest effect of interest.
  - `distill` is better than `pinn` at 90% and 99%.
  - The EMG + accelerometers set is equivalent at 99% and favours `pinn` at 90%.
  - Details in the README and on the website.
  - An independent recomputation from the raw records reproduces the 99% tests exactly.
  - **Teacher check:** max |difference| 0.85° (99%, fold 1, seed 0). The configuration, seed
    and mask are the same as study 3's run, and the differences go both ways; reported as is.
  - **Display changes on the study-4 page, after the result, with no effect on any number:**
    - each `base` vs `pinn` line shows the headroom, and "equivalent" is flagged where the
      headroom is below 0.5° (as the plan states);
    - the chosen λ per fold is listed, with the smallest-λ note;
    - the EMG + accelerometers 99% line is labelled "main share of this secondary set"
      instead of "primary";
    - the gyro-check subset lines no longer show the all-people means next to an n = 16
      test.
