#!/usr/bin/env bash
# x86 / A100 server environment (override Jetson defaults in environment.sh)
set -euo pipefail

export TensorRT_Lib="${TensorRT_Lib:-/usr/lib/x86_64-linux-gnu}"
export TensorRT_Inc="${TensorRT_Inc:-/usr/include/x86_64-linux-gnu}"
export TensorRT_Bin="${TensorRT_Bin:-/usr/src/tensorrt/bin}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_Lib="${CUDA_Lib:-$CUDA_HOME/lib64}"
export CUDA_Inc="${CUDA_Inc:-$CUDA_HOME/include}"
export CUDA_Bin="${CUDA_Bin:-$CUDA_HOME/bin}"
export CUDNN_Lib="${CUDNN_Lib:-/usr/lib/x86_64-linux-gnu}"

export DEBUG_PRECISION=fp16
export USE_Python=OFF
export ConfigurationStatus=Success

export PATH="$TensorRT_Bin:$CUDA_Bin:$PATH"
export LD_LIBRARY_PATH="$TensorRT_Lib:$CUDA_Lib:$CUDNN_Lib:${LD_LIBRARY_PATH:-}"

if [[ -f tool/cudasm.sh ]]; then
  # shellcheck disable=SC1091
  source tool/cudasm.sh
  export CUDASM="${cudasm:-86}"
else
  export CUDASM="${CUDASM:-86}"
fi

echo "CUDA_HOME=$CUDA_HOME  CUDASM=$CUDASM"
echo "TensorRT_Bin=$TensorRT_Bin"
