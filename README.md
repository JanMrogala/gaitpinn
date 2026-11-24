# gaitpinn

**Question:** two models, a data-only network (`base`) and the same network with a physics loss
(`pinn`), learn to predict hip, knee and ankle angles from EMG and IMU signals during level
walking. Which one keeps a low error for longer when the person starts climbing stairs?

- `PLAN.md`: question, split, metrics, threshold, decisions log.
- `DATA.md`: dataset comparison and the one chosen (Camargo et al. 2021).
- Code: `src/gaitpinn/`.
  - The two physics terms live in `gyro.py` and `swing.py`.
  - `matfile.py` reads the MATLAB tables.
  - `configs/data.yaml` records the data settings, whose values are constants in `data.py`.

## Run

```sh
uv venv -p 3.12 .venv && uv pip install -e '.[dev]'   # local: tests, plots, site
make hooks                                              # ruff + fast tests before each commit
make check
```

On LUMI (all settings come from `.env.local`, which is not committed):

```sh
jobs/sync.sh                  # copy the current clean commit to project storage
jobs/setup_lumi.sh            # once: container copy + venv
jobs/download.sh              # once: Camargo et al. 2021 into scratch
jobs/submit.sh prepare        # unzip + convert to one HDF5 file per person (dev-g)
jobs/submit.sh tune           # 2 models x 22 people x 12 configs, walking validation only
jobs/fetch.sh                 # copy run records back to ../runs, GPU-hour ledger
python -m gaitpinn.test select --runs ../runs     # chosen config per model
# copy ../runs/chosen_*.json to scratch, then:
jobs/submit.sh final          # 2 models x 22 people x 5 seeds + test stream
jobs/fetch.sh
python -m gaitpinn.test summary --runs ../runs    # numbers, Wilcoxon test, curves
python -m gaitpinn.plot --runs ../runs            # figures
make site                                         # website into ../site
```

Study 2 (`PLAN2.md`) uses the same steps with `prepare-x`, `xtune`, `xcal` and `xfinal`, and
`python -m gaitpinn.test xselect` / `xsummary`.

## Results

**No clear difference.** After the switch to stairs, `pinn` did not keep a lower joint-angle
error than `base`:

- **Error after switch** (RMSE over the stairs part, mean of hip, knee and ankle, n = 16
  people): `base` 11.38°, `pinn` 11.35°.
- **Test:** Wilcoxon p = 0.74, rank-biserial r = +0.10, Hodges–Lehmann
  difference +0.05° (95% CI -0.29 to +0.43).
- **All 22 people** (sensitivity check, not the verdict): p = 0.90.
- **Time to fail:** both models leave their walking error band within about 2 s of the switch
  (median 1.9 s for `base` and 1.8 s for `pinn` at the knee).

Excluded by the pre-registered gyro check: AB06, AB07, AB08, AB16, AB19, AB20.

Every number above comes from `runs/summary/results.json`, which `gaitpinn.test summary` builds
from the 220 final runs (`final-<model>-<person>-s<seed>-j23860765_<task>`, 2 models × 22
people × 5 seeds). The website shows the plots and per-person run IDs.

### Added after the main result: a strongly physics-constrained model

Not pre-registered: requested after the result above was known. `pinn_strong` is the tuned
`pinn` network with a fixed, large physics weight (λ = 0.1, chosen from walking-only loss
scales). It was compared with `base` by the same test on the same 16 people.

- **Error after switch:** `pinn_strong` 12.14°, `base` 11.38°. Wilcoxon p = 0.044,
  r = -0.57, Hodges–Lehmann difference (base − pinn_strong) -0.77° (95% CI -1.46 to
  -0.09).
- **Where the difference is:** mostly the knee (17.8° vs 15.9°, p = 0.011), on both climbing up and
  climbing down.
- **Reading:** strong physics made stair errors slightly worse, not better. This is a single
  added comparison that was not pre-registered, so treat p ≈ 0.04 as suggestive.
- **Runs:** `final-pinn_strong-<person>-s<seed>-j23861743_<task>` (110 runs).

### Limitations

- **Weak physics:** walking-validation tuning chose the smallest physics weight tried
  (λ = 0.001, the edge of the grid), so `pinn` is close to `base` by construction. The strongly
  constrained `pinn_strong` (above) was added afterwards to cover the other end.
- **Swing model:** the simple swing pendulum from the plan explains real knee swing poorly.
  Even with the true angles, its residual is about 2.6× the std of `I·knee_acc`.
- **Time to fail:** both models cross the threshold about 2 s after the switch, before the
  first stair step. This number mostly reflects the join between recordings and the walk up to
  the stairs.
- **Walking error `W`:** for some people (AB17, AB18, AB27, AB28), `W` is inflated by stretches
  of broken IK angles in level-ground test files that the quality rule did not catch. Their
  knee error in the stream's walking part is only 1–3°. This raises the threshold for both
  models alike.
- **Sample:** n = 16 after the pre-registered gyro-check exclusions. The result on all 22 is
  reported alongside.

## Study 2: one model for many people

Plan: `PLAN2.md`. One network is trained on many people (walking, ramps and stairs) and tested
on people it never saw, in 4 folds so that every person is tested once. `pinn_cal` is `pinn`
self-calibrated to each new person on 2 minutes of their unlabelled walking, using only the
physics losses.

**No clear difference in any of the three pre-registered tests** (error after switch, mean of
hip, knee and ankle, n = 22, Wilcoxon, Holm-corrected):

| Test | Mean error | p | Holm p | r | Hodges–Lehmann (95% CI) |
|---|---|---|---|---|---|
| `base` vs `pinn` | 7.31° vs 7.30° | 0.588 | 1.000 | +0.14 | +0.04° (-0.13 to +0.20) |
| `base` vs `pinn_cal` | 7.31° vs 7.36° | 0.702 | 1.000 | -0.10 | -0.03° (-0.20 to +0.15) |
| `pinn` vs `pinn_cal` | 7.30° vs 7.36° | 0.159 | 0.478 | -0.35 | -0.08° (-0.16 to +0.03) |

For `pinn` vs `pinn_cal`, one person (AB16) has exactly zero difference: the label-free gate
switched off the gyro term, the only calibration term in that fold. The Wilcoxon test drops
that person, so this test uses 21 people.

- **Physics during training:** tuning chose the smallest λ (0.001) in every fold. The
  validation score got worse as λ grew, so the final `pinn` is close to `base`.
- **Self-calibration:** it did not help on the new people. If anything it made the knee
  slightly worse (7.70° vs 7.47° for `pinn`; p = 0.05, one of about 65 uncorrected secondary
  tests, so suggestive only). The control `base_cal` shows the same pattern (7.37°).
- **Per fold:** folds 0–2 are within ±0.32°. In fold 3, `base` is better by a median of
  0.45° (vs `pinn`) and 0.66° (vs `pinn_cal`).
- **The 16 people who passed study 1's gyro check:** the same picture (all p > 0.27).

Every number comes from `runs/summary/results2.json` (`gaitpinn.test xsummary`), built from
the 440 final records `xfinal-<model>-<person>-s<seed>-j23872835_<task>` (4 models × 22
people × 5 seeds). The website's Study 2 page shows the plots, per-fold choices and the
calibration diagnostic.

**Limitations:**

- People in a fold share trained models, so the 22 pairs are not fully independent.
- The configuration and calibration setting can differ per fold.
- Input normalization uses all of a person's walking train files (sensor signals only), not
  just the 2-minute calibration walk.
- Training targets include the same IK glitches as study 1.
- **Run-to-run noise:** on the GPU, the same configuration and seed does not give the same
  network. Retraining a fold's chosen configuration at the tuning seed changed its validation
  score by up to 0.54°, and the 5 final seeds spread by 0.3–0.8°. The per-fold choices of lr,
  width and weight decay are therefore partly noise. The λ ordering was the same in all four
  folds. The final numbers average 5 seeds.

## Study 3: gaps in the true angles

Plan: `PLAN3.md`. As in study 2, but a share of the training and validation people's true
angles is removed in random 2 s blocks, while their sensors stay complete. `base` learns only
from the remaining angles. `pinn` also applies its physics loss to the stretches without angles.
Every person keeps a fully labelled 30 s calibration trial; test people are untouched.

**`base` is better at every share**, and the gap grows with the share (error after switch, mean
of hip, knee and ankle, n = 22, Wilcoxon, Holm-corrected over 50%, 90% and 99%):

| Missing | `base` | `pinn` | Holm p | Hodges–Lehmann `base − pinn` (95% CI) | Verdict |
|---|---|---|---|---|---|
| 0% (reference) | 7.53° | 7.69° | – (p = 0.70) | −0.05° (−0.42 to +0.13) | – |
| 50% | 7.60° | 7.80° | 0.030 | −0.15° (−0.34 to −0.03) | `base` better |
| 90% | 7.68° | 7.89° | 0.004 | −0.21° (−0.34 to −0.09) | `base` better |
| 99% | 8.27° | 8.81° | < 0.001 | −0.49° (−0.74 to −0.31) | `base` better |

- **Size:** the differences are small, below the pre-registered 0.5° smallest effect of
  interest except at 99%, where they reach it. Most of the 99% difference is at the knee
  (8.98° vs 10.21°).
- **Unlabelled stretches:** `pinn_lab` (physics only on chunks with angles) is no different from
  `pinn` at any share (p ≥ 0.52), so the physics loss on the stretches without angles neither
  helped nor hurt.
- **Trend:** the Page test for a `pinn` advantage that grows with the share is not significant
  (p = 0.91); the difference runs the other way.
- **Label efficiency:** `base` loses only 0.7° on stairs going from all angles to 99% missing
  (about 1 minute of angles per training person, half of it the calibration trial).
- **Tuning:** validation chose the smallest physics weight (λ = 0.001) in 15 of 16 share × fold
  cells.
- **The 16 people who passed study 1's gyro check:** same direction (p = 0.08, 0.03 and
  < 0.001 at 50%, 90% and 99%).

**Why:** the network already sees the gyros as inputs, so the gyro term adds no information,
and the swing pendulum is a poor model of real knee swing. Study 4 removes the gyros from the
inputs to test this.

Every number comes from `runs/summary/results3.json` (`gaitpinn.test gsummary`), built from the
1210 final records `gfinal-<model>-s<share>-<person>-s<seed>-j<job>_<task>` (arrays 23965276
and 23967891).

**Limitations:** gaps hit all modes equally (real occlusion is worse on stairs); people in a
fold share trained models; GPU runs are not bit-reproducible.

## Study 4: gyros only in the physics loss

Plan: `PLAN4.md`. As in study 3, but the network does not see the gyros. `base` and `pinn` get
EMG only (primary) or EMG + accelerometers (secondary). The gyros enter only `pinn`'s physics
loss (knee and ankle gyro term, no swing term). This matches a real use: train in a lab with
IMUs, then deploy a cheaper device without gyros. `distill` (secondary, EMG only) uses the gyros
another way: an all-sensor teacher (study 3's `base`) fills in the missing training angles.

**At 99% missing (EMG only), the pre-registered primary test favours `pinn`, but the effect is
small, and `distill` does better without any physics** (error after switch, mean of hip, knee
and ankle, n = 22, Wilcoxon):

| Missing | `base` | `pinn` | `distill` | Headroom (95% CI) | Holm p | Hodges–Lehmann `base − pinn` (95% CI) | Verdict |
|---|---|---|---|---|---|---|---|
| 0% (reference) | 8.86° | 9.03° | – | – | – (p = 0.10) | −0.13° (−0.31 to +0.01) | – |
| 50% | 9.12° | 9.07° | 9.02° | +0.26° (+0.14 to +0.41) | 0.633 | +0.02° (−0.06 to +0.11) | equivalent within ±0.5° |
| 90% | 9.43° | 9.62° | 9.17° | +0.57° (+0.40 to +0.73) | 0.275 | −0.13° (−0.33 to +0.02) | equivalent within ±0.5° |
| 99% (primary) | 10.80° | 10.40° | 9.98° | +1.94° (+1.59 to +2.27) | 0.003 | +0.34° (+0.15 to +0.58) | `pinn` better |

Headroom is how much `base` loses at that share compared with 0% (`base`@share − `base`@0%).
The Holm family for 50% and 90% is separate from the single primary test at 99%.

- **Size:** +0.34° is below the pre-registered 0.5° smallest effect of interest, and its CI
  includes 0.5°. `pinn` wins back about a fifth of what `base` loses from missing angles at 99%
  (0.34° of 1.94°).
- **Folds:** the per-fold medians of `base − pinn` are +0.75°, +0.03°, +0.08° and +0.30°. All
  four favour `pinn`, as the verdict rule requires, but two are near zero, and the largest is
  fold 0, the fold the feasibility pilot used (validation people only).
- **Per joint at 99%:** the gain is at the knee (+0.58°, p = 0.011) and the ankle (+0.51°,
  p < 0.001), the two joints with a gyro term. The hip is unchanged (p = 0.46).
- **`distill` beats `pinn`** at 90% (Holm p = 0.007, `distill − pinn` −0.35°) and 99% (Holm
  p = 0.006, −0.37°), and they are equivalent at 50%. `distill` is untuned and its teacher sees
  more (all 35 channels, including the accelerometers, and every training person's gyros).
  Still, by the plan's own reading, physics is not needed to make use of the extra sensors.
- **Where `pinn` is worse or flat:** at 0% `pinn` is 0.13° worse (p = 0.10; hip p = 0.046,
  uncorrected), and at 90% it is 0.13° worse (equivalent). The `base − pinn` difference grows
  with the share (Page trend p = 0.002).
- **Tuning:** the physics weight λ chosen per fold at 99% was 0.1–0.3. The smallest value,
  0.01 ("validation preferred almost no physics"), won only at 0% (2 of 4 folds).
- **The 16 people who passed study 1's gyro check (99%):** same direction (+0.41°, p = 0.005).

**EMG + accelerometers (secondary): the 99% result does not replicate.**

| Missing | `base` | `pinn` | Headroom (95% CI) | Holm p | Hodges–Lehmann `base − pinn` (95% CI) | Verdict |
|---|---|---|---|---|---|---|
| 0% (reference) | 7.42° | 7.44° | – | – (p = 0.19) | −0.07° (−0.16 to +0.03) | – |
| 50% | 7.51° | 7.49° | +0.08° (−0.01 to +0.16) | 0.656 | +0.03° (−0.11 to +0.15) | equivalent within ±0.5° |
| 90% | 7.75° | 7.47° | +0.32° (+0.16 to +0.48) | < 0.001 | +0.27° (+0.15 to +0.37) | `pinn` better |
| 99% | 8.69° | 8.48° | +1.26° (+1.00 to +1.52) | 0.353 | +0.12° (−0.13 to +0.45) | equivalent within ±0.5° |

- At 99%, the per-fold medians range from −0.41° to +0.94°.
- The 90% gain is a secondary test where the headroom is only 0.32°, so there was little to
  recover. It does not confirm the primary result.
- λ = 0.01 won in 1 of 4 folds at 90% and in 2 of 4 at 99%.

**Reading:** with the gyros removed from the inputs, the gyro physics term does carry some
information into a gyro-free network when almost all training angles are missing. The gain is
small, it sits at the knee and ankle, and a plain teacher–student approach (`distill`) uses the
same extra sensors better. This is the first of the four studies in which `pinn` beat `base`
on a pre-registered test (studies 1 and 2: no clear difference; study 3: `base` better).

**Checks:**
- An independent recomputation from the raw run records, without the project's analysis code,
  reproduces the means, statistics, p values, Hodges–Lehmann estimates and per-fold medians of
  the 99% tests (both input sets, and `distill` vs `pinn`).
- **Teacher check:** the `distill` teacher's validation score differs from study 3's `base` run
  with the same configuration, share, fold, seed and mask by 0.13° on average.
  - The maximum is 0.85° (99%, fold 1, seed 0). Differences go both ways (the next largest is
    −0.62°).
  - This is above study 2's 0.54° run-to-run spread. It is consistent with GPU non-determinism
    on the very small 99% training sets, but that was not proven.

Every number comes from `runs/summary/results_g4.json` and `results_g4a.json`
(`gaitpinn.test gsummary --prefix g4|g4a`), built from the final records
`g4final-<model>-…` and `g4afinal-<model>-…` (arrays 24008564, 24009038 and 24009276).

**Limitations:**
- The 22 test people are the same as in studies 1–3, whose results motivated this study.
- The pilot used fold 0's validation people.
- There is no placebo control (e.g. a time-shifted gyro).
- People in a fold share trained models.
- GPU runs are not bit-reproducible.
