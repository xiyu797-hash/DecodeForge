#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v compute-sanitizer >/dev/null 2>&1; then
  echo 'ERROR: compute-sanitizer missing (comes with CUDA toolkit). GPU memory safety unverified.' >&2
  exit 1
fi
if [[ ! -x build/gpu/decodeforge_gpu_tests ]]; then
  echo 'ERROR: build/gpu/decodeforge_gpu_tests not found. Run verify_gpu.sh first.' >&2
  exit 1
fi
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/gpu/decodeforge_gpu_tests
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/gpu/decodeforge_gpu_tests
if [[ ! -x build/gpu/decodeforge_ops_gpu_tests ]]; then
  echo 'ERROR: missing decodeforge_ops_gpu_tests' >&2
  exit 1
fi
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/gpu/decodeforge_ops_gpu_tests
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/gpu/decodeforge_ops_gpu_tests
echo '[SANITIZER] CUDA memcheck/racecheck PASS for both GPU test binaries'
