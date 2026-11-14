#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build/asan -DDF_ENABLE_CUDA=OFF -DDF_ENABLE_ASAN=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build/asan -j "$(nproc)"
ctest --test-dir build/asan --output-on-failure
