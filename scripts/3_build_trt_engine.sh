#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${ROOT}/runtime"
ONNX="${ONNX:-$ROOT/model/pointpillar_v2.onnx}"
PLUGIN="${RUNTIME}/build/libpointpillar_core.so"
PLAN="${PLAN:-$ROOT/model/pointpillar_v2.plan}"

if [[ ! -f "$ONNX" ]]; then
  echo "Missing $ONNX — run scripts/1_export_onnx.sh first"
  exit 1
fi
if [[ ! -f "$PLUGIN" ]]; then
  echo "Missing $PLUGIN — build runtime first (scripts/2_setup_runtime.sh + cmake/make)"
  exit 1
fi

TRTEXEC="${TensorRT_Bin:-}/trtexec"
if [[ ! -x "$TRTEXEC" ]]; then
  TRTEXEC="$(command -v trtexec || true)"
fi
if [[ -z "$TRTEXEC" ]]; then
  echo "trtexec not found. Install TensorRT and set TensorRT_Bin in environment.sh"
  echo "ONNX is ready at: $ONNX"
  exit 1
fi

mkdir -p "${ROOT}/model"
"$TRTEXEC" \
  --onnx="$ONNX" \
  --fp16 \
  --plugins="$PLUGIN" \
  --saveEngine="$PLAN" \
  --inputIOFormats=fp16:chw,int32:chw,int32:chw \
  --verbose 2>&1 | tee "${ROOT}/model/trt_v2_build.log"

echo "Saved TensorRT engine: $PLAN"
