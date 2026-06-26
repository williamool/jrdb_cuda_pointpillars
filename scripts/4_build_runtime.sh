#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME="${ROOT}/runtime"

if [[ ! -d "$RUNTIME" ]]; then
  echo "Run scripts/2_setup_runtime.sh first"
  exit 1
fi

cd "$RUNTIME"
# shellcheck disable=SC1091
source tool/environment_x86.sh
export CUDASM="${CUDASM:-80}"   # A100 = sm_80

mkdir -p build
cd build
cmake .. && make -j"$(nproc)"
echo "Built: $RUNTIME/build/pointpillar"
echo "       $RUNTIME/build/libpointpillar_core.so"
