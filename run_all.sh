#!/usr/bin/env bash
# Full benchmark in balanced blocks of 20 repetitions (resumable: finished cells are skipped).
# Block 1 = repetitions 0-19 for every cell, then the OULAD leave-one-module-out contrast,
# then blocks 2-5 (repetitions 20-99). Any completed block leaves a balanced design.
set -u
cd "$(dirname "$0")"
PY=./.venv/Scripts/python.exe
[ -x "$PY" ] || PY=./.venv/bin/python
JOBS=${JOBS:-12}
mkdir -p logs
$PY -I src/lsb/run_experiment.py --rep-start 0 --reps 20 --jobs "$JOBS" --out results/raw  > logs/run_block1.log 2>&1 || exit 1
$PY -I src/lsb/run_experiment.py --views "oulad_*" --lomo --rep-start 0 --reps 20 --jobs "$JOBS" --out results/lomo > logs/run_lomo1.log 2>&1 || exit 1
for start in 20 40 60 80; do
  $PY -I src/lsb/run_experiment.py --rep-start $start --reps 20 --jobs "$JOBS" --out results/raw > logs/run_block_$start.log 2>&1 || exit 1
done
$PY -I src/lsb/run_experiment.py --views "oulad_*" --lomo --rep-start 20 --reps 80 --jobs "$JOBS" --out results/lomo > logs/run_lomo2.log 2>&1 || exit 1
echo "all blocks finished"
