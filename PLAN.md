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

- **Dataset:** Camargo et al. 2021 (see `DATA.md`): 22 people, one session each.
- **Target:** right-leg sagittal hip, knee and ankle angles from the motion-capture IK table
  (`hip_flexion_r`, `knee_angle_r`, `ankle_angle_r`), in degrees.
- **Inputs:** right-leg sensors only.
  - **EMG:** 11 channels. Rectified, then a causal low-pass filter (envelope).
  - **IMU:** 24 channels, accelerometer and gyro of the foot, shank, thigh and trunk units.
  - **Never used as inputs:** goniometers, markers, force plates, inverse dynamics.
- **Rate:** everything is resampled to 100 Hz. Every filter is causal.
- **Normalization:** mean and std per channel, from that person's walking **training** files
  only.
- **Storage:** one HDF5 file per person on LUMI scratch.

### Split (per person)

- **Walking files:** all `treadmill` and `levelground` files.
  - They are split **by file** (never by window): 70% train, 15% validation, 15% walking test.
  - The split is stratified so each part holds both treadmill and level-ground files.
  - Files are kept whole, including the stand and turn spans inside level-ground circuits.
  - In step 4 I check that the validation and test treadmill speeds fall inside the training
    speed range.
- **Stair files:** all `stair` files are **test only**.
- **Ramp files:** not used.
- **No stair data** may touch training, tuning, normalization or sensor alignment. A test checks
  this.

### Test stream (per person)

The stream has two parts:

1. **Walking part:** the first 30 s of the person's level-ground walking test files, joined in
   file order.
2. **Stairs part:** all of the person's stair files, back to back, in recorded order.
   - **Recorded order:** within a parent recording, by `trialStarts`; parent recordings by
     file name. Camargo gives no absolute clock across parent recordings.

**`t = 0`** is the first sample of the first stair file. Each Camargo stair file starts with
2–3 s of walking up to the stairs. It then contains the ascent, a 3–6 s stand and turn on the
top platform, and the descent.

The models see the stream as one continuous signal, as a wearable device would.

## Models

- **Network:** one small causal network, the same for both models. A causal 1D CNN
  (6 dilated convolution layers, kernel 3, dilations 1–32) sees the last 127 samples
  (1.27 s) of inputs.
- **Output:** hip, knee and ankle angles at the current sample. `pinn` also outputs knee
  torque, used only by the swing term (not trained against the dataset's inverse-dynamics
  torques).
- **Training settings:** identical for both: optimizer, epochs, early stopping on walking
  validation, batch size.
- **One model per person and seed.**

### `pinn` physics loss

`pinn` loss = angle error + λ × (gyro term + swing term). Each term lives in its own file
with a test.

**1. Gyro term** (knee: thigh and shank IMUs; ankle: shank and foot IMUs)

- Joint angle speed comes from the two segment gyros.
- **Alignment:** per person, fitted on walking **training** files only. A least-squares
  projection of both gyros (3 axes each) onto `d(true angle)/dt`.
- The term compares that projected speed with `d(predicted angle)/dt`.
- **Hip has no gyro term:** Camargo has a trunk IMU, not a pelvis IMU.

**2. Swing term** (knee)

- During swing, the shank and foot are one pendulum hanging from the moving thigh:
  `I·knee_acc = muscle_torque − gravity_torque − k·(knee − knee0) − b·knee_speed`.
- **Muscle torque:** `muscle_torque = Σ w_m · a_m`.
  - `a_m` is the EMG envelope passed through a first-order activation filter.
  - Each `w_m ≥ 0` is learned, with a fixed sign per muscle (flexor or extensor).
- **Body segments:** mass, length and inertia come from standard body-segment tables, scaled
  by each person's height and weight (`SubjectInfo.mat`).
- **Derivation:** the equation is derived with sympy.
- **Swing detection:** from IMU signals only, never from force plates.

**Required physics tests**

- The swing term is about 0 on a simulated pendulum with known torque.
- The gyro term is small when computed with the **true** angles on real walking data.

### Tuning (same effort for both)

- **Configurations:** both models get the same fixed number, 12 each.
  - Shared: learning rate {1e-3, 3e-3} × width {32, 64}.
  - `base` adds weight decay {0, 1e-5, 1e-4}.
  - `pinn` adds λ {0.03, 0.3, 3}. The physics terms are in normalized units.
- **Data:** walking validation data only. One tuning seed (not one of the 5 final seeds), all
  22 people.
- **Choice:** one configuration per model, the one with the lowest mean walking-validation
  RMSE across people. It is then used for every person and seed.
- **Budget:** tuning runs count towards the GPU budget.

## Metrics

All metrics are per person, model, seed and joint, in degrees.

- **Rolling error:** RMSE over a **trailing** 2 s window, as a function of time in the stream.
- **Walking error `W`:** RMSE over the person's whole walking-test split. The same `W` is used
  for the dashed line on the plot and for the failure threshold.
- **Error after switch:** RMSE over the stairs part (all samples with `t ≥ 0`).
- **Time to fail:** the first `t ≥ 0` at which the rolling error is above 1.5 × `W` and stays
  above it for 2 s. If that never happens, it is reported as "> end of stream" (the stream
  length).
- **Seeds:** 5 per model and person. Results are averaged over seeds first, so each person
  counts once.

### Plots

**Main plot (`error_over_time`)**, as specified:

- x = time since switch (s), y = rolling error (°).
- One panel each for hip, knee and ankle.
- `base` and `pinn` lines show the mean over people, with a 95% band across people.
- A vertical line at `t = 0` and a dashed line at each model's mean `W`.

Because people are at different phases (climbing, standing, descending) at the same `t`, this
mean blurs phases together.

**Secondary plot (`error_by_bout`)**, pre-registered:

- Error against time since **each stair entry**: the first sample labelled `walk-stairascent`
  or `walk-stairdescent`, from −2 s to +6 s.
- Per-sample RMSE is averaged over a person's ~40 bouts, then over people. There is no 2 s
  window.
- Labels are used only to align the scoring, never as model inputs.

### Comparison and verdict

- **Primary endpoint:** error after switch, averaged over hip, knee and ankle.
  - The question is about all three joint angles. Hip has no physics term, so a knee-only
    endpoint would tilt the test towards `pinn`.
- **Test:** a paired, two-sided Wilcoxon signed-rank test across the 22 people, on
  `base − pinn`.
- **Effect size:** the matched-pairs rank-biserial correlation, plus the median paired
  difference in degrees with a 95% bootstrap interval.
- **Verdict:**
  - "`pinn` is better": p < 0.05 and median `base − pinn` > 0.
  - "`base` is better": p < 0.05 and median `base − pinn` < 0.
  - Otherwise "no clear difference".
  - All three outcomes are reported equally clearly.
- **Secondary results** (reported, not part of the verdict): per-joint error after switch, time
  to fail per joint (same paired test), and the `error_by_bout` curves.

## Compute (LUMI)

What was actually used, checked on the system on 2026-11-19:

- **Container:** `lumi-multitorch-full-u24r72f21m50t211-20261028_090706.sif` from
  `/appl/local/laifs/containers`, sha256 `3ebb506cd91e1160744bd9a6933bd918638e6080543eec2090fca821aead0cb6`.
  - Contents: PyTorch 2.11.0 (ROCm 7.2), Python 3.12, plus scipy, h5py, sympy, plotly and pytest.
  - It is copied to project storage.
  - A venv (`--system-site-packages`) on top holds only this package.
- **Modules:** `module purge && module load Local-LAIF lumi-aif-singularity-bindings`.
- **Partitions:** `dev-g` for tests (time limit shown as 3 h, not the 30 min in older notes);
  `small-g` for real runs. Each run uses 1 GCD and a Slurm job array over
  (model × person × seed). CPU partitions are not used.
- **Billing:** 0.5 GPU-hour per GCD-hour, if a job uses ≤ 8 cores and ≤ 64 GB per GCD. GPU-hours
  per run are taken from `sacct`.
- **Budget:**
  - **Cap:** the project's remaining GPU-hours (967 on 2026-11-19), re-read with
    `lumi-allocations` before each batch.
  - **Ask first:** before any batch over 20% of what remains.
  - **Time:** each batch of jobs must finish within 24 h of submission, including queue time.
- **MIOpen:** `MIOPEN_USER_DB_PATH` and `MIOPEN_CUSTOM_CACHE_DIR` point to a job-local folder
  under `/tmp`.

## Decisions log

- **2026-11-19, stairs files opened during dataset selection.** Before any model existed,
  single stair files of AB06, AB25 and AB30 were opened. The purpose was to check labels,
  durations and whether `d(knee)/dt` matches the gyro difference. Findings:
  - Stair phases last about 1–3 s.
  - The gyro sagittal axis and sign differ per person.

  No setting in this plan is tuned on those files. Sensor alignment is fitted from walking
  training files only.
- **2026-11-19, walking data.** The training task is level walking: treadmill plus level-ground
  circuits. Ramps are not used. Agreed with the user.
- **2026-11-19, one model per person and seed.** Agreed with the user.
