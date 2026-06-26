#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAMPLE="${1:-024653}"
DATA_ROOT="${DATA_ROOT:-/home/wbl/OpenPCDet/data/jrdb}"
BIN_DIR="${ROOT}/data/sample_bins"
OUT_DIR="${ROOT}/data/trt_output"

mkdir -p "$BIN_DIR" "$OUT_DIR"

if [[ -f "$BIN_DIR/${SAMPLE}.bin" ]]; then
  echo "Using bundled sample: $BIN_DIR/${SAMPLE}.bin"
else
  export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
  python3 "$ROOT/tool/prepare_bin.py" --sample_ids "$SAMPLE" --out_dir "$BIN_DIR" --data_root "$DATA_ROOT"
fi

cd "$ROOT/runtime/build"
./pointpillar "$BIN_DIR/" "$OUT_DIR/" --timer
echo "Predictions in $OUT_DIR"
