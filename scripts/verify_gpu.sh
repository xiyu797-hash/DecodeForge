#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v nvcc >/dev/null 2>&1; then
  echo 'ERROR: nvcc not found. Install CUDA Toolkit 12+ before GPU verification.' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo 'ERROR: nvidia-smi not found. No NVIDIA GPU runtime verified.' >&2
  exit 1
fi
# 如须规避其他用户的0号卡，在外部设置 CUDA_VISIBLE_DEVICES=1。
cmake -S . -B build/gpu -DDF_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/gpu -j "$(nproc)"
ctest --test-dir build/gpu --output-on-failure
# CTest中必须出现GPU测试；CPU-only误配置不能被当作真机验收。
ctest --test-dir build/gpu -N | grep -q 'gpu_correctness' || { echo 'ERROR: gpu_correctness test not registered' >&2; exit 1; }
ctest --test-dir build/gpu -N | grep -q 'gpu_ops_correctness' || { echo 'ERROR: gpu_ops_correctness test not registered' >&2; exit 1; }
./build/gpu/decodeforge_cli device
./build/gpu/decodeforge_cli gpu-check --n 133 --k 1025 --batch 8 --group 64
./build/gpu/decodeforge_ops_cli gemm-check --m 37 --n 65 --k 129 --dtype fp16
./build/gpu/decodeforge_ops_cli gemm-check --m 37 --n 65 --k 129 --dtype bf16
./build/gpu/decodeforge_ops_cli norm-check --rows 31 --cols 1025
mkdir -p benchmarks
./build/gpu/decodeforge_cli bench-all --repeats 100 --warmup 20 --csv benchmarks/gpu_benchmark.csv
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 512 --dtype fp16 --csv benchmarks/fp16_gemm.csv
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 512 --dtype bf16 --csv benchmarks/bf16_gemm.csv
./build/gpu/decodeforge_ops_cli norm-bench --rows 128 --cols 1024 --csv benchmarks/rmsnorm.csv

# 额外GPU运算完整性和性能烟测（数据仅为当前设备的实际值）
./build/gpu/decodeforge_ops_cli gemm-check --m 17 --n 33 --k 65 --dtype fp16
./build/gpu/decodeforge_ops_cli gemm-check --m 17 --n 33 --k 65 --dtype bf16
./build/gpu/decodeforge_ops_cli norm-check --rows 7 --cols 1025
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 256 --dtype fp16 --csv benchmarks/gemm_fp16.csv
./build/gpu/decodeforge_ops_cli gemm-bench --m 256 --n 256 --k 256 --dtype bf16 --csv benchmarks/gemm_bf16.csv
./build/gpu/decodeforge_ops_cli norm-bench --rows 256 --cols 1024 --csv benchmarks/norm.csv
echo '[GPU VERIFY] ALL CUDA CHECKS PASS (see CTest and CSV)'
