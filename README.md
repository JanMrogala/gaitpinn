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
