#!/usr/bin/env bash
# Held-out runs (scenes 100-299, both datasets) with the parameters chosen on dev
# (results/dev_tuning/chosen.json), then the fresh-noise Pfa verification.
# Usage: bash run_heldout.sh [workers]
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
W=${1:-10}
export HDF5_USE_FILE_LOCKING=FALSE
SPECS=$($PY -c "
import json; c=json.load(open('results/dev_tuning/chosen.json')); L=c['L']
s=['agm L=%d'%L['agm'],'john L=%d'%L['john'],'cond L=%d'%L['cond'],'cond_twc L=%d'%L['cond'],'cond_zl L=%d'%L['cond'],
   'med L=%d'%L['med'],'tkeo_stft kt=%d'%c['kt'],'tkeo_sb']
if L['cond']!=8: s+=['cond L=8','cond_twc L=8']
print(';'.join(s))")
IFS=';' read -ra ARR <<< "$SPECS"
for ds in impaired clean; do
  if [ $ds = clean ]; then D='D:\data\wideband\scenes_clean.h5'; else D='D:\data\wideband\scenes.h5'; fi
  for spec in "${ARR[@]}"; do
    set -- $spec
    P=""; [ -n "$2" ] && P="--param $2"
    name=$($PY -c "import sys; sys.argv=['x']; from run_scenes import make; print(make('$1', ['$2'] if '$2' else []).name)")
    [ -f results/${name}_${ds}/metrics.json ] && continue
    $PY run_scenes.py run --det $1 $P --data "$D" --start 100 --workers $W --out results/${name}_${ds} 2>&1 | tail -1
  done
done
for spec in "${ARR[@]}"; do
  set -- $spec
  P=""; [ -n "$2" ] && P="--param $2"
  $PY run_scenes.py verify --det $1 $P 2>&1 | tail -1
done
