# gaitpinn

**Question:** two models, a data-only network (`base`) and the same network with a physics loss
(`pinn`), learn to predict hip, knee and ankle angles from EMG and IMU signals during level
walking. Which one keeps a low error for longer when the person starts climbing stairs?

- `PLAN.md`: question, split, metrics, threshold, decisions log.
- `DATA.md`: dataset comparison and the one chosen (Camargo et al. 2021).

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
