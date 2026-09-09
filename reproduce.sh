#!/usr/bin/env bash
# Reproduces every table and figure in the paper, in order.
# Usage:  bash reproduce.sh


set -e
export PYTHONPATH="$PWD/src/utils:$PWD:$PYTHONPATH"

echo "=== 1/6  Generate dataset ==============================="
python3 data/synthetic/generate_dataset.py \
        --output data/synthetic/tasks.csv

echo "=== 2/6  Train and save the IDCRA GBT =================="
python3 src/utils/train_and_save_gbt.py

echo "=== 3/6  Classification results (Tables III, VII) ======"
python3 src/idcra/evaluate_idcra.py

echo "=== 4/6  Stability check (Table VIII, Fig. 9) =========="
python3 src/utils/stability_check.py

echo "=== 5/6  Scheduling sweep (Tables IX, X) ==============="
echo "         [slow: Naive Fog dominates the runtime]"
python3 src/utils/run_full_sweep.py Cloud-Only Threshold-Fog HEFT-Fog CF-SAF
python3 src/utils/run_full_sweep.py "Naive Fog"

echo "=== 6/6  Power comparison (Table XI) ==================="
python3 src/utils/run_30seed_power.py 0 10
python3 src/utils/run_30seed_power.py 10 20
python3 src/utils/run_30seed_power.py 20 30

echo "=== Report all tables =================================="
python3 src/utils/report_tables.py

echo
echo "Figures: python3 src/utils/capture_figdata.py"
echo "         python3 src/utils/make_figures.py"
echo "Outputs in results/ and results/figures/"
