#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPCDET_ROOT="${OPENPCDET_ROOT:-/home/wbl/OpenPCDet}"
CKPT="${CKPT:-/home/wbl/OpenPCDet/output/jrdb_models/pointpillar_v2/jrdb_v2_bs8/ckpt/checkpoint_epoch_100.pth}"
CFG="${CFG:-$ROOT/configs/pointpillar_jrdb_v2.yaml}"
OUT_DIR="${OUT_DIR:-$ROOT/model}"

# OpenPCDet training env (Python 3.10 + compiled pcdet ops)
PYTHON="${PYTHON:-/root/2TStorage/envs/pointcept-utonia/bin/python}"
export OPENPCDET_ROOT
export PYTHONPATH="${OPENPCDET_ROOT}:${ROOT}:${PYTHONPATH:-}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python not found: $PYTHON"
  echo "Set PYTHON=/path/to/your/conda/env/bin/python"
  exit 1
fi

"$PYTHON" -m pip install -q onnx onnxsim onnx-graphsurgeon 2>/dev/null || true

mkdir -p "$OUT_DIR"
echo "Exporting PointPillar v2"
echo "  CKPT: $CKPT"
echo "  CFG:  $CFG"
echo "  OUT:  $OUT_DIR"

cd "$OPENPCDET_ROOT/tools"
"$PYTHON" "$ROOT/tool/export_onnx.py" \
  --cfg_file "$CFG" \
  --ckpt "$CKPT" \
  --out_dir "$OUT_DIR" \
  --onnx_name pointpillar_v2

echo "Done:"
ls -lh "$OUT_DIR"/pointpillar_v2*.onnx
