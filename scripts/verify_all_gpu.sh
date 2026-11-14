#!/usr/bin/env bash
# 真正的GPU最终验收：必须在可用的NVIDIA显卡上执行。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p benchmarks
{
  echo "date_utc=$(date -u +%FT%TZ)"
  echo '--- GPU + DRIVER ---'
  nvidia-smi -L
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,temperature.gpu,utilization.gpu --format=csv
  echo '--- TOOLCHAIN ---'
  nvcc --version
  cmake --version | head -1
  g++ --version | head -1
} > benchmarks/device_environment.txt
bash scripts/verify_cpu.sh
bash scripts/verify_gpu.sh
bash scripts/verify_gpu_compute_sanitizer.sh
echo '[RELEASE] NVIDIA GPU qualification PASS; environment recorded in benchmarks/device_environment.txt'
