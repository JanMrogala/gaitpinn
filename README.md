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

Not yet available.
