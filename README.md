# CF-SAF: Cloud-Fog Smart Agriculture Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Code, dataset generator and measurement pipeline for:

> **Energy-Efficient Fog–Cloud Framework with Sensor Data Classification and
> Resource Scheduling for Smart Agriculture IoT Systems**
> Sneha, Prabh Deep Singh, Vikas Tripathi, G.L. Saini
> Graphic Era (Deemed to be University), Dehradun · Manipal University Jaipur

Every table and figure in Sections VI–VII of the paper is regenerable from
this repository.

---

## Quick start

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` runs the whole pipeline in order and prints each table.

**Runtime.** Steps 1–4 take a few minutes. Step 5 is slow: the Naive Fog
baseline routes all traffic to the fog tier and scales roughly
quadratically in task count, so it needs ~20 min per seed (~3.5 h for 10
seeds). `run_full_sweep.py` takes system names and resumes from
`results/full_results.json`, so it can be run in pieces:

```bash
python3 src/utils/run_full_sweep.py Cloud-Only HEFT-Fog CF-SAF   # fast
python3 src/utils/run_full_sweep.py "Naive Fog"                  # slow
```

To skip the slow baseline entirely and still verify the central claim, run
only steps 1–3 and 6. Step 6 (Table XI) is the CF-SAF vs HEFT-Fog
comparison that isolates the scheduler.

---

## Expected output

Small deviations from the values below are normal — thread scheduling and
BLAS non-determinism affect the discrete-event simulator. The dataset
generator and the classifier are fully deterministic and should match
exactly.

### Deterministic (should match exactly)

`python3 data/synthetic/generate_dataset.py`

```
Urgent  : 14388 (28.8%)
Duration: 1005.8 s over 1200 devices
Naive observed-tau rule agrees with label: 92.59%   (100% => leakage)
Grouped split: train=35512 val=6960 test=7528
```


`python3 src/idcra/evaluate_idcra.py` → **paper Tables III and VII**

```
F1: theta*=0.22 P=0.837 R=0.938        <- Table III, row F1
F2: theta*=0.07 P=0.811 R=0.948        <- Table III, row F2
F3: theta*=0.05 P=0.726 R=0.967        <- Table III, row F3

Class 0: P=0.971 R=0.916 F1=0.943 FNR=0.084 n=5228    <- Table VII
Class 1: P=0.831 R=0.938 F1=0.881 FNR=0.062 n=2300    <- Table VII
Macro  : P=0.901 R=0.927 F1=0.912
Weight : P=0.928 R=0.923 F1=0.924
```

### Stochastic (should be close, not identical)

`python3 src/utils/stability_check.py` → **paper Table VIII**

Drift below ~1% up to 270 tasks/s, then a sharp jump. The exact drift
figure at 280 varies between runs because the system is unstable there —
that instability is the finding, not noise to be suppressed.

| Rate | Paper drift | Expected |
|---|---|---|
| 250 | 0.8% | < 2% |
| 260 | 0.7% | < 2% |
| 270 | 1.0% | < 2% |
| 280 | 147.9% | ≫ 50% |

`python3 src/utils/report_tables.py` → **paper Tables IX, X, XI**

| System | Paper $\bar{L}$ (ms) | Tolerance |
|---|---|---|
| Cloud-Only | 202.44 | ±2 |
| Naive Fog | 9580.90 | ±1500 |
| Threshold-Fog | 343.81 | ±40 |
| HEFT-Fog | 272.79 | ±60 |
| **CF-SAF** | **223.36** | ±35 |


```
vs Naive Fog       -97.7%   p ~ 1e-11    significant
vs Threshold-Fog   -35.0%   p ~ 1e-07    significant
vs HEFT-Fog         -9.9%   p ~ 7e-04    significant  (30 seeds, Table XI)
vs Cloud-Only      +10.3%   p ~ 5e-02    Cloud-Only lower
```

### Figures

```bash
python3 src/utils/capture_figdata.py    # writes results/figdata/*.pkl
python3 src/utils/make_figures.py       # writes results/figures/*.png
```

---

## Layout

```
config/config.yaml              All hyperparameters and node classes
data/synthetic/
  generate_dataset.py           Dataset generator + grouped_split()
  tasks.csv                     Pre-generated (regenerates identically)
data/raw/dataset_sources.md     DOIs for the three real datasets
src/idcra/                      IDCRA: features, GBT, calibrator, eval
src/orsa_f/                     ORSA-F scheduler
src/simulation/simulator.py     Discrete-event simulator, energy model
src/utils/
  paths.py                      Repo-relative paths (import, don't hardcode)
  train_and_save_gbt.py         Step 2 — trains results/models/gbt.pkl
  stability_check.py            Table VIII
  run_full_sweep.py             Tables IX, X   (resumable)
  run_30seed_power.py           Table XI       (resumable)
  report_tables.py              Prints Tables IX, X, XI
  capture_figdata.py            Collects per-task/per-node data
  make_figures.py               Figures 2–9
results/                        Stored outputs from the reported runs
reproduce.sh                    Runs everything in order
FINDINGS.md                     Record of corrections made during review
```


## Citation

```bibtex
@article{sneha2025cfsaf,
  title   = {Energy-Efficient Fog--Cloud Framework with Sensor Data
             Classification and Resource Scheduling for Smart
             Agriculture IoT Systems},
  author  = {Sneha and Singh, Prabh Deep and Tripathi, Vikas
             and Saini, G. L.},
  journal = {IEEE Internet of Things Journal},
  year    = {2025},
  note    = {Under review}
}
```

MIT licensed — see `LICENSE`.
