# Changelog

## Unreleased (2026-11-13)

- Protect invalid Shape arithmetic, reject partially parsed CLI integers, and reject GPU output downloads before a successful execution or Graph replay. Prepare two-pass workspaces before CUDA Event timing even with zero warmup.
- For Direct only, use compile-time division or power-of-two shift for Scale indexing according to K; keep the Grouped Paired and Scalar kernels unchanged. Add threshold, signed INT4, group boundary, odd tail, Graph and autotune GPU regressions.
- Add opt-in nine-pair autotuner internal trace and separate subsequent holdout samples, Unicode-safe Windows paths and exclusive raw sample files. Preserve the existing default tuning and session cache semantics.
- Add reproducible Windows matrix, balanced A/B and analysis scripts; preserve failed, noisy and A/A experiments. Fix per-shape AB/BA ordering and document the earlier ordering confound.
- Make Windows sanitizer checks use a verified temporary drive alias when needed and validate NVIDIA tool output, exit codes and error summaries. Current build qualification and limitations: `RESUME_READINESS.md`.

## 1.0.0-rc1 (2026-11-09)

- Retained grouped scalar-nibble reference as `GroupedScalar` alongside paired packed INT4 grouped kernel.
- Added 7 paired alternating-order timing samples and median speedup for scalar-vs-paired ablation.
- GPU correctness tests exercise four independent W4A16 paths, including graph capture; autotune correctness checked on two representative batch shapes rather than re-tuning every case.
- Enabled user-selected CUDA architectures, with SM89 default for RTX 4060 / RTX 4090.
- Fixed event constructor partial-allocation cleanup and duplicate sanitizer runs.
- Added isolated Windows scripts for a completed-build sanitizer run and a 12-shape ablation benchmark.
- Expanded packed INT4 CPU regression invariants and added transparent qualification status.

CPU build/tests validated on Linux GCC, Clang, Clang ASan/UBSan. CUDA GPU qualification pending.
