# Second revision: corrected simulator and new experiments

This update accompanies the second revision of the CF-SAF manuscript.

## What changed

**`src/simulation/simulator.py`: preemption bug fixed.** Earlier releases had two errors in `_try_preempt`:

1. A suspended task's pending completion event was not cancelled, so the task was counted as completed twice and its CPU share was released twice.
2. An arriving task could be queued twice.

The corrected version cancels the event, saves the suspended task's progress (2 ms state-save overhead, `EPS_PRE_MS`), and suspends only a task whose release actually makes room for the arriving task.

**`revision2/`: scripts for every result in the revised manuscript**, at the 240 tasks/s operating point. The stored outputs are in `revision2/results/`.

The scripts in `src/utils/` (270 tasks/s) belong to the previous version of the paper and are superseded by `revision2/`.

## Setup (once)

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD/src/utils:$PWD"
python3 data/synthetic/generate_dataset.py --output data/synthetic/tasks.csv
python3 src/utils/train_and_save_gbt.py
```

For the real-data experiments, download the three `stuard_*.csv` files from Mendeley Data (doi:10.17632/35wh56287y.2) into `data/real/`.

## Rebuild tables and figures from the stored results (fast)

```bash
cd revision2
python3 tables.py       # prints every number used in Section VIII
python3 make_figs.py    # writes Figs. 3-11 to revision2/figures/
```

## Re-run the experiments (optional, slow)

Run from inside `revision2/`. Each script overwrites its file in `results/`.

| Manuscript result | Script | Approx. time |
|---|---|---|
| Stability test (Table V, Fig. 3) | `python3 stab_fix.py` | 10 min |
| Main comparison (Tables VII-VIII, Figs. 4-8) | `python3 regen_main.py` | 30 min (Naive Fog is the slow part) |
| 30-seed power, ablation, theta sweep, scalability (Tables IX-X, Figs. 9-10) | `python3 regen_rest.py`, then `python3 ablation240.py` | 40 min |
| Cloud-Only comparison and cloud-size sweep (Table XI) | `python3 exp_cloud.py` | 15 min |
| Rebalancing scenarios S1-S4 (Table XII, Fig. 11) | `python3 exp_rebalance.py` | 15 min |
| Lemma 2 / Corollary 3 checks (Sec. VIII-I) | `python3 boundcheck.py` | 5 min |
| Independent and real workloads (Tables XIII-XIV) | `python3 exp_indep.py`, then `python3 exp_indep_sched.py` | 15 min |

The simulator is deterministic for a given seed, so re-running reproduces the stored values exactly.
