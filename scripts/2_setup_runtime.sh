#!/usr/bin/env bash
# Copy NVIDIA CUDA-PointPillars sources and apply JRDB PointPillar v2 patches.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_CUDA_PP="${CUDA_POINTPILLARS_ROOT:-/home/wbl/CUDA-PointPillars}"
RUNTIME="${ROOT}/runtime"
PATCHES="${ROOT}/runtime_patches"

if [[ ! -d "$SRC_CUDA_PP/src" ]]; then
  echo "CUDA-PointPillars not found at $SRC_CUDA_PP"
  exit 1
fi

rm -rf "$RUNTIME"
mkdir -p "$RUNTIME"
rsync -a --exclude build "$SRC_CUDA_PP/" "$RUNTIME/"

cp "$PATCHES/main.cpp" "$RUNTIME/src/main.cpp"
cp "$PATCHES/lidar-postprocess.hpp" "$RUNTIME/src/pointpillar/lidar-postprocess.hpp"
cp "$PATCHES/lidar-postprocess.cu" "$RUNTIME/src/pointpillar/lidar-postprocess.cu"
cp "$PATCHES/environment_x86.sh" "$RUNTIME/tool/environment_x86.sh"

echo "Runtime prepared at $RUNTIME (PointPillar v2 patches applied)"
echo "Next:"
echo "  cd $RUNTIME && . tool/environment.sh && mkdir -p build && cd build && cmake .. && make -j"
