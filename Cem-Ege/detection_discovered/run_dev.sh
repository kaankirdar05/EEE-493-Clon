#!/usr/bin/env bash
# Dev tuning on scenes 0-99. Usage: bash run_dev.sh <impaired|clean> [workers]
# Skips configurations whose metrics.json already exists.
cd "$(dirname "$0")"
PY=../.venv/Scripts/python.exe
ds=$1; W=${2:-12}
export HDF5_USE_FILE_LOCKING=FALSE
if [ $ds = clean ]; then D='D:\data\wideband\scenes_clean.h5'; else D='D:\data\wideband\scenes.h5'; fi
for spec in "agm L=4" "agm L=8" "agm L=16" "agm L=32" "john L=4" "john L=8" "john L=16" "john L=32" \
            "cond L=4" "cond L=8" "cond L=16" "cond L=32" "med L=4" "med L=8" "med L=16" "med L=32" \
            "tkeo_stft kt=4" "tkeo_stft kt=8" "tkeo_stft kt=16" "tkeo_sb"; do
  set -- $spec
  P=""; [ -n "$2" ] && P="--param $2"
  name=$1_${2//=/}; name=${name%_}
  [ -f results/dev_tuning/${name}_${ds}/metrics.json ] && continue
  $PY run_scenes.py run --det $1 $P --data "$D" --start 0 --scenes 100 --examples 0 --workers $W \
      --out results/dev_tuning/${name}_${ds} 2>&1 | tail -1
done
