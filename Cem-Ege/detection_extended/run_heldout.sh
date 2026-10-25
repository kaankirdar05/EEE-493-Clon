#!/usr/bin/env bash
# Part A held-out runs (scenes 100-299) with the parameters chosen on dev scenes 0-99.
# Usage: bash run_heldout.sh [workers]
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
W=${1:-10}
for ds in impaired clean; do
  if [ $ds = clean ]; then D='D:\data\wideband\scenes_clean.h5'; else D='D:\data\wideband\scenes.h5'; fi
  for spec in "cav L=4" "acorr L=32" "entropy_spec m=64" "entropy_hist bins=16" "scd_fam Np=16" "cav_pw L=4" "entropy_spec_pw m=64"; do
    set -- $spec
    name=$($PY -c "import sys; sys.argv=['x']; from run_partA import make; print(make('$1', ['$2']).name)")
    $PY run_partA.py run --det $1 --param $2 --data "$D" --start 100 --workers $W --out results/${name}_${ds} 2>&1 | tail -1
  done
done
$PY compare_partA.py > /dev/null
