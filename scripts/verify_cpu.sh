#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build/cpu -DDF_ENABLE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpu -j "$(nproc)"
ctest --test-dir build/cpu --output-on-failure
./build/cpu/decodeforge_cli cpu-check --n 23 --k 129 --batch 4 --group 64
# CPU-only必须拒绝GPU调用，而不是隐式以CPU代替。
if ./build/cpu/decodeforge_cli gpu-check --n 16 --k 16 --batch 1 --group 16 >/dev/null 2>&1; then
  echo 'ERROR: CPU-only build unexpectedly passed gpu-check' >&2; exit 1
fi
if ./build/cpu/decodeforge_ops_cli gemm-check --m 1 --n 1 --k 1 >/dev/null 2>&1; then
  echo 'ERROR: CPU-only build unexpectedly passed CUDA GEMM' >&2; exit 1
fi
echo '[CPU VERIFY] ALL PASS (including fail-closed GPU gates)'
