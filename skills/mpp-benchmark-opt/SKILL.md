---
name: mpp-benchmark-opt
description: Add or extend benchmark programs in this NPP repository, or use benchmark data to drive optimization. Use when Codex needs to generate benchmark files with `benchmark/scripts/new_benchmark.sh`, implement usable benchmark bodies, build and run `benchmark/`, collect MPP or NVIDIA baselines, compare CSV results, and close the loop with verification.
---

# MPP Benchmark Workflow

Use the repository's existing benchmark toolchain as documented in the project docs. Do not rely on older benchmark skills in this repo as the source of truth.

## Load only what you need

- Read [`docs/project/mpp_benchmark_and_optimization_guide.md`](../../docs/project/mpp_benchmark_and_optimization_guide.md) first. It is the repo-specific workflow source of truth.
- Read [`benchmark/README.md`](../../benchmark/README.md) for CLI usage and output files.
- Read [`benchmark/FRAMEWORK.md`](../../benchmark/FRAMEWORK.md) only when you need framework internals, executable selection, generator behavior, or compare semantics.

## Follow this workflow

When the task is to add a new benchmark program, Codex should directly create and complete the benchmark file. Do not stop at analysis or a stub unless the API semantics are genuinely blocked.

1. Confirm correctness is already stable in `build` and, when needed, `build-nvidia`.
2. Define the benchmark target and scope explicitly: API/module, reference side, input sizes, and expected deliverables.
3. If a benchmark file does not exist yet, run `benchmark/scripts/new_benchmark.sh` first to generate the scaffold in the correct module directory.
4. Replace the generated `success=false` stub with a real benchmark body:
   - organize ROI / signal length
   - allocate buffers
   - compute step / pitch
   - call the real API
   - fill `BenchmarkResult`
   - compute throughput
5. Fix the benchmark scenario before collecting numbers: mode, filter, fixed runtime constants, executable source, output format, and output prefix.
   Randomized inputs must use the framework's fixed seed helpers in `benchmark/benchmark_base.h`; do not introduce ad-hoc `rand()` initialization.
6. Build benchmark binaries with `benchmark/scripts/build_benchmark.sh` unless a matching benchmark binary already exists and is intentionally reused.
7. Run the new benchmark on MPP first and confirm it produces real CSV rows.
8. When needed, run the same scenario with `--use-nvidia-npp`.
9. Compare results with `benchmark/scripts/compare_csv.py`, then identify the hotspot and write one optimization hypothesis.
10. If the task includes optimization, change `src/` code, rerun focused regression tests, rerun the exact same benchmark scenario, then generate a progress report when the iteration spans before/after/reference.
11. Finish only when the result clearly states: benchmark file path, verification status, measured gain or regression when applicable, and next action.

## Generate benchmark files directly

Use the generator first instead of manually creating an empty `.cpp` file:

```bash
./benchmark/scripts/new_benchmark.sh <type> <category> <header> \
  --functions f1,f2[,f3...] \
  [--variant-tags tag1,tag2]
```

Example:

```bash
./benchmark/scripts/new_benchmark.sh nppi addc nppi_arithmetic_and_logical_operations.h \
  --functions nppiAddC_8u_C1RSfs,nppiAddC_32f_C1R \
  --variant-tags baseline
```

After generation, Codex must keep going and implement the real benchmark logic in the generated file. A generated stub alone does not count as finished.

## Use the existing scripts

Preferred build commands:

```bash
./benchmark/scripts/build_benchmark.sh
./benchmark/scripts/build_benchmark.sh --use-nvidia-npp

./build.sh --benchmark
./build.sh --benchmark --use-nvidia-npp
```

Preferred run commands:

```bash
./benchmark/scripts/run_benchmark.sh -f Resize -o resize_mpp
./benchmark/scripts/run_benchmark.sh --use-nvidia-npp -f Resize -o resize_nvidia
```

Preferred compare / progress commands:

```bash
python3 benchmark/scripts/compare_csv.py \
  resize_mpp.csv \
  resize_nvidia.csv \
  -o resize_compare.csv \
  --html resize_compare.html \
  --print-table \
  --filter Resize

python3 benchmark/scripts/progress_report.py \
  mpp_before.csv \
  mpp_after.csv \
  nvidia_ref.csv \
  -o mpp_progress.csv \
  --html mpp_progress.html
```

## Guardrails

- Never compare runs with different filter, fixed runtime constants, input sizes, or benchmark executable source.
- Never compare runs if one side changed the benchmark input randomization rule or bypassed the fixed seed in `benchmark/benchmark_base.h`.
- Treat `mpp_time_pct_of_nvidia = (mpp_avg_time_ms / nvidia_avg_time_ms) * 100` as the source-of-truth compare rule.
- Interpret `< 100%` as MPP faster, `> 100%` as NVIDIA faster.
- Do not invent a second benchmark framework or ad-hoc standalone runner when the current `benchmark/` framework can express the API.
- Do not hand-create benchmark files from scratch before trying `benchmark/scripts/new_benchmark.sh`.
- Do not leave a generated benchmark in `success=false` stub state if the API semantics are already clear enough to implement.
- Keep `benchmark/scripts/build_benchmark.sh`, `benchmark/scripts/run_benchmark.sh`, `benchmark/scripts/compare_csv.py`, and `benchmark/scripts/progress_report.py` as separate layers; do not invent a new combined compare flow in `benchmark/scripts/run_benchmark.sh`.
- Check whether `scripts/run_benchmark.sh` is using the expected `build*/benchmark/npp_benchmark`.
- Do not mix multiple unrelated optimizations into one benchmark iteration.
- Do not report a performance gain without rerunning relevant functional or precision tests.
- Prefer changing only `benchmark/` and docs when extending benchmark capability; do not widen changes into main project config unless benchmark requirements cannot be expressed with current interfaces.

## Deliverables

- Benchmark file path and implemented API list
- Explicit benchmark scenario: mode, binary source, filter, fixed runtime constants, output names
- Saved baseline / reference CSVs
- Compare CSV or HTML report
- Progress CSV or HTML report for before/after iterations when applicable
- Explicit regression-test status
- Written conclusion stating benchmark implementation status, hotspot when applicable, current gain, remaining gap, and next step

## Finish only when

- benchmark inputs and executable source are explicit
- the benchmark file is actually created or updated in `benchmark/`
- baseline and reference CSVs are saved
- compare results are reproducible
- regression-test status is explicit
- the result states the hotspot, the gain or regression, and the next action
