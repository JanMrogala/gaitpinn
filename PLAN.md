# Plan

Written before any model was trained. If anything changes after stairs results have been seen,
it is logged at the bottom with the reason.

## Question

Two models are trained on **level walking only**. Each is then fed a stream that switches from
walking to **stair climbing**, with no retraining. Which model keeps a low joint-angle error for
longer after the switch?

| Name | What it is |
|---|---|
| `base` | small causal network, trained on angle error only (data-only) |
| `pinn` | the same network, trained on angle error plus a physics loss |

**Main result:** a plot of error over time since the switch, with `base` and `pinn` on the same
axes.

## Data

- **Dataset:** Camargo et al. 2021 (see `DATA.md`). 22 people, one session each.
- **Target:** right-leg sagittal hip, knee and ankle angles (degrees), taken from the
  motion-capture IK table: `hip_flexion_r`, `knee_angle_r`, `ankle_angle_r`.
- **Inputs:** right-leg sensors only.
  - **EMG:** 11 channels.
  - **IMU:** 24 channels, the accelerometer and gyro of the foot, shank, thigh and trunk units.
  - **Never used as inputs:** goniometers, markers, force plates, inverse dynamics, labels.
- **Processing:** one rate, 100 Hz, and causal filters only. Each recorded file is processed on
  its own, with filter state starting at zero.
  - **EMG (1000 Hz):** rectify, then a 2nd-order Butterworth low-pass at 6 Hz, then keep every
    10th sample.
  - **IMU (200 Hz):** a 2nd-order Butterworth low-pass at 40 Hz, then keep every 2nd sample.
  - **Target (200 Hz):** keep every 2nd sample. It is never an input.
  - **First 0.5 s of each file:** used as model context but never scored or used in a loss, so
    filter start-up does not count.
- **Normalization:** per-channel mean and std from that person's walking **training** files
  only.
- **Storage:** one HDF5 file per person on LUMI scratch.

### Split (per person, fixed)

- **Walking files:** all `treadmill` and `levelground` files.
  - **By file, never by window,** separately for each mode, with split seed 0. Validation gets
    `max(1, floor(0.15·n + 0.5))` files, walking test the same number, and training the rest.
  - **One split per person,** shared by `base`, `pinn`, the tuning seed and all 5 final seeds.
  - Files are kept whole, including the stand and turn spans inside level-ground circuits.
- **Stair files:** test only.
- **Ramp files:** not used.
- **Checked in step 4, before anything is trained:**
  - At least 3 files of each mode.
  - Validation and test treadmill speeds fall inside the training speed range. If not, files are
    swapped between parts so they do, and that is logged.
  - At least 30 s of level-ground walking test data. If less, use what there is, and log it.

### No leakage

Stair data never touches training, tuning, normalization or **any calibration**:

- sensor alignment;
- the swing detector and its thresholds;
- the physics constants and their starting values;
- the filter settings;
- early stopping and configuration choice.

All of these come from walking training files only, or are fixed in this plan. A test checks
that the file IDs feeding each of these steps contain no stair file.

### Exclusion (decided before any stairs prediction)

A person is excluded if, on walking data only, any of these holds:

- an EMG or IMU channel is missing or flat;
- there are fewer than 3 files of either walking mode;
- the gyro check (below) still fails after the alignment is fixed.

Each exclusion is logged with its reason. Everything else is reported for every included person.

### Test stream (per person)

The stream has two parts:

1. **Walking part:** the first 30 s of the person's level-ground walking **test** files,
   joined in file order.
2. **Stairs part:** the person's stair files in recorded order. Within a parent recording the
   order follows `trialStarts`; parent recordings are ordered by file name. Camargo has no
   absolute clock across parent recordings.

`t = 0` is the switch from part 1 to part 2.

- The models see the stream as one continuous signal, as a wearable device would. Their
  context and the rolling error window run across file joins without reset.
- The stairs part is the **whole stair files**, joined back to back. Each holds an approach,
  the ascent, a stand and turn on the top platform, the descent and walk-off. Chosen by the user.

## Models

- **Network:** one small causal 1D CNN, the same for both models. It has 6 dilated convolution
  layers (kernel 3, dilations 1–32) and sees the last 127 samples (1.27 s).
- **Output:** hip, knee and ankle angles at the current sample. `pinn` also outputs a knee
  torque, used only by the swing term (not trained against inverse-dynamics torques).
- **Training:** both models use identical settings.
  - Adam optimizer, 4 s chunks drawn from within single files, batch size 64.
  - At most 60 epochs; early stopping with patience 8.
  - Early stopping and configuration choice use the **walking-validation angle RMSE, averaged
    over hip, knee and ankle**, for both models.
- **One model per person and seed.**

### `pinn` physics loss

`pinn` loss = angle MSE + λ × (gyro term + swing term). Each term lives in its own file and has
a test. All three are made unit-free, so that λ means the same for every person:

- **Angle MSE:** computed on normalized angles.
- **Gyro term:** the residual is divided by the std of `d(true angle)/dt` in walking training
  data.
- **Swing term:** each residual is divided by the std of `I·knee_acc` from the true knee angle
  during swing in walking training data.

**1. Gyro term** (knee: thigh and shank IMUs; ankle: shank and foot IMUs)

- **Sensor alignment:** each sensor gets one unit-length sagittal axis, from walking **training**
  files.
  - The axis is the first principal axis of that sensor's gyro signal during walking.
  - Its sign is chosen so that `axis_distal·g_distal − axis_proximal·g_proximal` correlates
    positively with `d(true angle)/dt`.
  - There is no free scale, so the term stays the spec's
    `d(knee)/dt = gyro_shank − gyro_thigh`.
- **Hip** has no gyro term: Camargo has a trunk IMU, not a pelvis IMU.
- **Required check:** with the **true** angles on walking **validation** files, both of these
  must hold for every person and joint:
  - correlation ≥ 0.85;
  - RMS of (`gyro speed − d(true angle)/dt`) ≤ 0.7 × RMS of `d(true angle)/dt`.

  With no free scale, a correlation of 0.85 and an amplitude ratio of about 1.15 already give a
  residual of about 0.6, so 0.5 would fail people whose correlation passes.

  If not, fix the alignment first. Then apply the exclusion rule.

**2. Swing term** (knee)

- **Model:** during swing, the shank and foot are one pendulum hanging from the moving thigh.
  The equation is derived with sympy, with these inputs:
  - **Body segments:** mass, centre of mass and inertia from standard body-segment tables
    (Winter), scaled by each person's height and mass from `SubjectInfo.mat`.
  - **Thigh orientation:** a causal complementary filter on the thigh IMU.
  - **Thigh angular speed and acceleration:** from the thigh gyro.
- **Two residuals, both used during swing only:**
  1. `I·knee_acc + gravity_torque(thigh, knee) + thigh-motion terms + k·(knee − knee0) +
     b·knee_speed − τ̂ = 0`
  2. `τ̂ − Σ w_m·a_m = 0`

  Here `τ̂` is the network's torque output, and `knee`, `knee_speed` and `knee_acc` come from
  the predicted knee angle by central finite differences. That is used in training only, so the
  model stays causal.
- **Learned, per model:**
  - `w_m ≥ 0` (softplus), with a fixed sign per muscle: flexor or extensor.
  - `k ≥ 0` and `b ≥ 0` (softplus).
  - `knee0`.

  All of these start from fixed values set in `configs/pinn.yaml`.
- **`a_m`:** the EMG envelope through a first-order activation filter with a fixed 40 ms time
  constant.
- **Swing detection:** from IMU signals only (shank gyro and foot accelerometer), never from
  force plates. Its thresholds are set on walking training files.

### Tuning (same effort for both)

- **Configurations:** both models get the same number, 12 each.
  - Shared: learning rate {1e-3, 3e-3} × width {32, 64}.
  - `base` adds weight decay {0, 1e-5, 1e-4}.
  - `pinn` adds λ {0.03, 0.3, 3}, with weight decay fixed at 0.
- **Data:** walking validation data only. One tuning seed (100), not one of the 5 final seeds
  (0–4), all included people.
- **Choice:** one configuration per model, the lowest mean walking-validation angle RMSE across
  people. It is then used for every person and seed.

## Metrics

All errors are RMSE in degrees, per person, model, seed and joint.

- **Rolling error:** RMSE over a **trailing** 2 s window.
- **Walking error `W`:** RMSE over the person's **level-ground** walking test files. This
  matches the walking part of the stream. One `W` is used for both the dashed line and the
  failure threshold. The RMSE over the whole walking test split, treadmill included, is
  reported separately.
- **Error after switch:** RMSE over the stairs part.
- **Time to fail:**
  - Computed per seed: the first `t ≥ 0` at which the rolling error rises above 1.5 × `W` and
    stays above it for the next 2 s.
  - A hold that runs past the end of the stream counts as not failed.
  - **Censoring:** a run that never fails is set to `H`, the stairs-part length of the person
    with the shortest stream. All time-to-fail values are also capped at `H`, so every person
    has the same horizon. A run that never fails is reported as "> H".
  - **Per person:** the median over the 5 seeds.
- **Seeds:** every other metric is averaged over the 5 seeds per person first, so each person
  counts once.

### Plots

**Main plot (`error_over_time`):**

- x = time since switch (s), from −30 s to `H`; y = rolling error (°).
- One panel each for hip, knee and ankle.
- Lines for `base` and `pinn`: the mean over people of the seed-averaged rolling error.
- The band is the t-based 95% confidence interval of that mean across people.
- A vertical line at `t = 0` and a dashed line at each model's mean `W`.

**Secondary plot (`error_by_bout`):**

- Error against time since each stair entry, plotted separately for **ascent** and **descent**.
  - Entries are the first sample labelled `walk-stairascent` or `walk-stairdescent`.
  - The window runs from −2 s to +6 s.
- **Formula:** at each offset, `sqrt(mean over bouts of e²)` per person, then the mean over
  people.
- Labels are used only to align the scoring, never as model inputs.
- In step 4, the descent entry label is checked on the label tables only, before any model is
  trained.

### Comparison and verdict

- **Primary endpoint:** error after switch, averaged over hip, knee and ankle. The question is
  about all three joint angles. Hip has no physics term, so a knee-only endpoint would tilt the
  test towards `pinn`.
- **Test:** a paired, two-sided Wilcoxon signed-rank test across included people, on
  `base − pinn`, with `zero_method="wilcox"`.
- **Effect size:** the matched-pairs rank-biserial correlation `r`, plus the Hodges–Lehmann
  estimate of `base − pinn` in degrees with a 95% bootstrap interval.
- **Verdict:**
  - "`pinn` is better": p < 0.05 and `r > 0`.
  - "`base` is better": p < 0.05 and `r < 0`.
  - Otherwise "no clear difference".
  - All three outcomes are reported equally clearly.
- **Secondary results** (reported, not part of the verdict):
  - error after switch per joint;
  - error after switch on ascent and on descent samples only;
  - time to fail per joint (same paired test, using the capped values);
  - the `error_by_bout` curves.

## Compute (LUMI)

What was actually used, checked on the system on 2026-11-19:

- **Container:** `lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif` from
  `/appl/local/laifs/containers`, sha256 `3ebb506cd91e1160744bd9a6933bd918638e6080543eec2090fca821aead0cb6`.
  It holds PyTorch 2.11.0 (ROCm 7.2), Python 3.12, scipy, h5py, sympy, pyyaml and plotly.
- **Modules:** `module purge && module load Local-LAIF lumi-aif-singularity-bindings`.
- **Paths** (`<project>` and `<user>` come from `.env.local`):
  - `/project/<project>/<user>/gaitpinn/`, holding:
    - `code/<commit>/`: one folder per synced commit, each with its `COMMIT` file;
    - `container/`: the copied image;
    - `venv/`: built with `--system-site-packages`.
  - `/scratch/<project>/<user>/gaitpinn/`, holding `data/`, `runs/`, `ckpt/` and `logs/`.
- **Partitions:**
  - `dev-g` for tests; the time limit shown on the system is 3 h, not the 30 min in older notes.
  - `small-g` for real runs: 1 GCD each, as a Slurm job array over (model × person × seed).
  - CPU partitions are not used: the user ruled them out, and the project's CPU allocation is
    already used up.
- **Billing:** 0.5 GPU-hour per GCD-hour, when a job uses ≤ 8 cores and ≤ 64 GB per GCD.
  - Jobs ask for 4 cores and 60 GB.
  - GPU-hours come from `sacct` for every job, failed ones included, and are kept in
    `runs/billing.csv` as a running total.
- **Budget:**
  - **Cap:** the project's remaining GPU-hours (967 on 2026-11-19), re-read with
    `lumi-allocations` before each batch.
  - **Estimate first:** each batch gets a GPU-hour estimate before submission, based on measured
    per-run times. Tuning runs count too.
  - **Ask first:** before any batch over 20% of what remains.
  - **Time:** the whole study, from data download to the final graph, must take under 24 h of
    wall-clock time. Tuning and seeds are sized to fit, measured per-run times first.
- **MIOpen:** `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` point to a job-local folder
  under `/tmp`.

## Decisions log

- **2026-11-19, stair files opened during dataset selection.** Before any model existed,
  single stair files of AB06, AB25 and AB30 were opened. The purpose was to check labels,
  durations and whether `d(knee)/dt` matches the gyro difference. Findings:
  - stair phases last about 1–3 s;
  - each file contains an approach, a stand and turn, and walk-off;
  - the gyro sagittal axis and sign differ per person.

  These findings shaped three choices:
  - the per-person sensor alignment;
  - the `error_by_bout` window (−2 to +6 s);
  - the question to the user about the stairs part.

  No number in this plan was tuned on those files. In step 4, the need for per-person alignment
  is re-checked on **walking training files only**, and the result is logged here.
- **2026-11-19, walking data.** The training task is level walking: treadmill plus level-ground
  circuits. Ramps are not used. Agreed with the user.
- **2026-11-19, one model per person and seed.** Agreed with the user.
- **2026-11-19, stairs part = whole stair files** (option A), chosen by the user before any
  model was trained. The alternative was to keep only the climbing segments.
- **2026-11-19, time limit:** the whole study must take under 24 h of wall-clock time. Set by
  the user, who will relax it if it limits the study too much.
