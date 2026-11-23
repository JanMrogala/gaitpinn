# Plan, study 3: gaps in the true angles

Written before any study-3 model was trained, and revised once after an independent review,
still before any training. Changes made after results have been seen are logged at the bottom
with the reason. Studies 1 and 2 (`PLAN.md`, `PLAN2.md`) and their results are not changed by
this study.

## Question

In practice, the true joint angles come from motion capture, and stretches of them are often
missing (markers hidden). The sensors (EMG, IMU) keep recording. A data-only model can learn
only from the stretches that have angles. A physics-informed model can also learn from the
stretches without them, because its physics loss needs no true angles.

**Question:** as more of the training angles are missing, does `pinn` keep a lower error than
`base` on new people?

| Name | What it is |
|---|---|
| `base` | study 2's network, trained on the remaining true angles only |
| `pinn` | the same network, trained on the remaining true angles plus the physics loss on all data, with and without angles |
| `pinn_lab` (secondary) | `pinn`'s chosen configuration with physics only on the labelled chunks, no unlabelled data. It shows whether a `pinn` gain comes from the stretches without angles. |

**Main result:** error on stairs against the share of missing angles, with `base` and `pinn`,
for people the models never saw. Plus error over time since the switch to stairs at each share.

## Setup (from study 2, unchanged unless stated)

- Same data (`data/h5x`), same 22 people, same 4 folds (training, validation, test people), same
  test stream (walking, then stairs) and same metrics as study 2.
- Same network, batch size, chunk length, steps per epoch, epoch cap and patience as study 2.
- No self-calibration (`pinn_cal`) in this study.
- **Exception, held fixed from study 2 for comparability:** the target-quality rule (which
  files exist for a person) was computed from the full angles when the data were converted. It
  is a file list, not a value fed to a model.

## Missing angles

- **Shares missing:** 0%, 50%, 90% and 99% of the samples outside the calibration trial (below).
  - 0% is a full run of the study-3 pipeline: the reference point, and secondary only.
- **Pattern:**
  - Every file is cut into consecutive 2 s blocks from its first sample; a shorter last block
    counts as one block.
  - Each block keeps its true angles or loses all of them. It is kept if its random number is
    below 1 − share.
  - One uniform random number per block comes from
    `SeedSequence([draw, crc32("person/file")])`.
  - Within a draw, the angles kept at 99% are a subset of those kept at 90%, and those at 90%
    are a subset of those kept at 50%.
- **Draws:** each run's seed is its mask draw: tuning seed 100 uses draw 100, and final seed s
  uses draw s.
  - `base` and `pinn` with the same seed see the same mask, so the comparison stays paired.
  - Averaging over the 5 final seeds also averages over 5 gap patterns.
- **Who loses angles:** training and validation people, in all their files (walking, ramps,
  stairs).
- **Test people are not affected.** Their true angles are only ever used for scoring, as in
  study 2.
- **Calibration trial:** every training and validation person keeps the true angles of one
  fixed 30 s stretch at every share.
  - It is the first 30 s of study 2's calibration walk: the first treadmill walking train file,
    from 10 s to 40 s, or level ground when there is no treadmill file.
  - It stands for the short, fully tracked calibration recording that a motion-capture session
    normally starts with.
  - Both models use it in their angle loss.
- **Reported before training,** from the masks only (no angles are read): per share and draw,
  the realised share, kept seconds per person and per mode (walking, ramps, stairs), and kept
  validation seconds per fold.
- **Limitation:** gaps hit all modes equally. Real occlusion is worse on stairs.

## Per-person setup: from the calibration trial only

Everything a training or validation person's physics setup needed from true angles in study 2
now comes from their calibration trial (first 15 s for fitting, last 15 s for checking):

- **Gyro alignment:**
  - principal axes from the gyros of all walking training files (sensors only, as before);
  - signs and the `lsq` fit on the first 15 s;
  - the `pca`/`lsq` choice and the gyro check (corr ≥ 0.85, relative residual ≤ 0.7) on the
    last 15 s.
- **Thigh sign and swing scale:** from the 30 s.
- **Swing threshold, EMG scaling and gyro scales:** from sensors, as in study 2.

This setup is the same at every share. The physics people per fold are reported and compared
with study 2's set, which was decided on study 1's validation files.

Every value computed from true angles (speeds, smoothing, scales) is computed inside the
calibration trial only. No such array is built across a gap.

## What reads the remaining angles

- **Angle loss:** kept samples only.
- **Target normalization:** global mean and std over kept samples of the training people.
- **Validation score** (early stopping and all choices): RMSE pooled over all kept, scored
  samples of the fold's validation people, mean of hip, knee and ankle.
  - The calibration trials guarantee some kept samples.
  - A fold with none is an error.
- **Implementation:** lost samples are set to NaN when loaded, losses mask with selection (not
  multiplication), and every loss and score is asserted to be finite.

**No-leak test:** on synthetic people whose gyros match their angles (so physics people
exist), the lost angles in a copy of the data are replaced once with random values and once
with ±1e6. A short CPU run must then give exactly the same:
- person setups and target statistics;
- every batch tensor entering a loss;
- per-step losses and validation scores;
- final weights.

## Training

- **Labelled batch (all models):** 64 chunks of 4 s.
  - Each chunk is built around one kept sample, drawn uniformly from all kept samples of the
    fold's training people (people weighted by kept seconds).
  - The chunk's start is drawn so that this sample lies in the scored part of the chunk, after
    the network's 1.26 s warm-up.
  - Chunks are clipped to the file, the file's first 0.5 s is not scored, and short files are
    padded, as in study 2.
  - Only kept samples count in the angle loss.
  - The batch comes from its own random stream, so `base` and `pinn` with the same seed see
    identical labelled batches.
- **Unlabelled batch (`pinn` only):** 64 more chunks from anywhere in the physics people's
  data, in proportion to their length, from a second random stream. It carries no angles.
- **Physics loss (`pinn`):** gyro and swing terms pooled over all samples of physics people in
  both batches, weighted by λ.
- **`pinn_lab`:** physics loss on the labelled batch only.
- The mean number of angle-loss samples per batch is logged.

## Tuning (same effort for both models)

- **Grid:** per share and fold, 16 configurations per model at seed 100.
  - Shared: lr {1e-3, 3e-3} × width {64, 128}.
  - `base` adds weight decay {0, 1e-5, 1e-4, 1e-3}.
  - `pinn` adds λ {0.001, 0.01, 0.1, 1}.
- **Choice:** per share and fold, the lowest validation score.
- If λ = 0.001 wins at a share, the result is reported as "validation preferred almost no
  physics".
- Validation score against λ and against weight decay is reported per share and fold.
- `pinn_lab` is not tuned; it uses `pinn`'s choice.

## Final runs, metrics and tests

- **Final runs:** per share, fold and model, the chosen configuration with seeds 0–4.
- **Metrics:** per test person, averaged over the 5 seeds, as in study 2.
- **Primary endpoint:** error after switch (RMSE over the stairs part), mean of hip, knee and
  ankle.
- **Primary tests:** one per share (50%, 90%, 99%), `base − pinn`, paired over the 22 people.
  - Two-sided Wilcoxon signed-rank (`zero_method="wilcox"`), Holm-corrected over the three.
  - Effect sizes: rank-biserial r and Hodges–Lehmann with a 95% bootstrap CI.
- **Smallest effect of interest:** 0.5°. Verdict per share:
  - **"better":** Holm p < 0.05 with r in that model's favour, and the per-fold median
    difference favouring the same model in at least 3 of 4 folds.
    - With fewer agreeing folds: "better (pooled only)".
  - **"equivalent within ±0.5°":** otherwise, if the Hodges–Lehmann 95% CI lies inside
    (−0.5°, +0.5°).
  - **"inconclusive":** otherwise.
- **Secondary results:**
  - 0% share;
  - `pinn` vs `pinn_lab`;
  - walking error and per joint;
  - climbing up and down;
  - the 16 people who passed study 1's gyro check;
  - per-fold medians;
  - physics people per fold;
  - whether the `base − pinn` difference grows with the share: a Page trend test over 50%,
    90% and 99%.
- **Limitation:** people in a fold share trained models, as in study 2. GPU runs are not
  bit-reproducible (study 2: up to 0.54° from the same seed).

## Compute

- **Tasks:**
  - tuning: 4 shares × 4 folds × 2 models × 16 configurations = 512;
  - final: 4 × 4 × 5 seeds × 2 models = 160, plus `pinn_lab` 3 × 4 × 5 = 60.
- **Estimate:** about 40 GPU-hours, timed on a pilot first.
- **Partition:** `small-g`, 1 GCD per task. Batches over 20% of the remaining budget are asked
  for first.

## Decisions log

- **2026-11-20, study 3 defined** with the user: version 1 of the gap experiment (missing
  true angles, complete sensors), on study 2's setup.
- **2026-11-20, plan revised after an independent review, before any code or training.**
  Changes:
  - fixed 30 s calibration trial for the per-person setup;
  - mask tied to the seed;
  - symmetric 4-value grids instead of an edge rule;
  - 0% run with the study-3 code;
  - `pinn_lab` arm;
  - smallest effect of interest and the fold-consistency condition;
  - pooled validation score;
  - sampler, NaN masking, separate random streams and the no-leak test specified.
- **2026-11-20, masks and pilot (job 23891411; tuning tasks 399 and 415, array 23891412,
  commit cdf9eb2). No final runs exist yet.**
  - **Masks, draw 0, median kept seconds per person:**

    | Missing | Kept | Kept on stairs |
    |---|---|---|
    | 0% | 3244 s | 547 s |
    | 50% | 1658 s | 286 s |
    | 90% | 360 s | 56 s |
    | 99% | 63 s (incl. the 30 s trial) | 6 s |

    - Realised shares: 49.7%, 89.8% and 99.0%.
    - At 99%, two people keep no stair angles.
    - Kept validation seconds per fold at 99%: 173–198 s.
  - **Calibration-trial setup:** the gyro check passes 20 of 22 people (all but AB10 and
    AB13), giving 11–13 physics training people per fold.
    - It differs from study 1's check on walking validation files, which passed 16 people.
    - The rule is the same at every share, so it does not bias the comparison.
  - **Timing:** `base` 7.5 min and `pinn` 11 min per task (`pinn` processes two batches per
    step).
    - Estimate for the remaining 510 tuning tasks: about 38 GPU-hours.
    - With a 40 min limit per task, at most 170 GPU-hours.
  - The pilot runs are genuine tuning runs; the full batch skips their indices.
  - The training code is unchanged since commit cdf9eb2; later commits touched only the
    summary, the plots and the website.
- **2026-11-23, tuning done (512 runs: arrays 23891412, 23892357, 23963697 and 23964460; the
  first chunk waited in LUMI's queue over the weekend). Choices frozen in `runs/chosen_g.json`
  before any final run.**
  - **λ:** 0.001 won in 15 of 16 share × fold cells; the exception is 99% missing, fold 0
    (0.01).
    - As pre-stated: validation preferred almost no physics.
    - In every cell, `pinn`'s best validation score is above `base`'s, at every share
      (0%: by 0.08–0.38°; 99%: by 0.13–0.69°).
  - **`base` weight decay:** varies; 1e-3 (the new top value) won at 90% fold 3 and at 99%
    folds 0 and 1.
  - Ledger after tuning: 33.1 GPU-hours in total for the project.
