---
name: mpp-api-implement
description: Implement, repair, and validate NPP APIs in this repository by matching NVIDIA NPP behavior as the baseline. Use when Codex needs to add a missing API, fix incorrect behavior, write or repair focused tests, validate semantics in `build-nvidia`, update local `src/` code, or refresh coverage after implementation or test changes.
---

# MPP Implementation Workflow

Use NVIDIA NPP as the functional reference unless the project docs already document an intentional deviation. Do not rely on other skills in this repo for implementation instructions.

## Load only what you need

- Read [`docs/project/mpp_api_implementation_guide.md`](../../docs/project/mpp_api_implementation_guide.md) for the repo-specific implementation workflow.
- Read [`test/README.md`](../../test/README.md) for test layout and naming.
- Inspect `API/`, `API-12.2/`, `api_analysis/api_functions_12.2.csv`, `api_analysis/coverage.csv`, nearby `src/`, and nearby `test/unit/` before editing code.

## Follow this workflow

1. Identify the exact API signature, data types, layout, ROI rules, and expected `NppStatus`.
2. Find the closest existing implementation and the closest existing test in the same module family.
3. Write or repair the focused test first. Cover normal path, invalid parameters, ROI/step behavior, and precision-sensitive behavior where relevant.
4. Build and run the focused test in `build-nvidia` so NVIDIA NPP validates the test itself.
5. Implement or repair the local `src/` code only after the test is trustworthy.
6. Run the same focused test in `build`.
7. Refresh coverage with `./update_coverage.sh` when implementation or test status changes.
8. Record what behavior was matched, what files changed, and what remains out of scope.

## Use the existing build and test flow

Reference commands:

```bash
./build.sh --use-nvidia-npp
cd build-nvidia
ctest --output-on-failure
./unit_tests --gtest_filter="*ApiOrModule*"

./build.sh
cd build
ctest --output-on-failure
./unit_tests --gtest_filter="*ApiOrModule*"

./update_coverage.sh
```

For quick module triage:

```bash
python3 tools/check_coverage.py
```

Then inspect `api_analysis/coverage.csv` and `api_analysis/coverage_summary.md` directly to locate the target API and confirm whether the task is an implementation gap or a test gap.

## Compare behavior explicitly

When behavior is unclear, compare these dimensions explicitly:

- returned `NppStatus`
- output values
- alpha preservation rules
- in-place behavior
- saturation, rounding, or clamping behavior
- ROI legality
- helper API behavior such as buffer-size queries

## Guardrails

- Do not implement first and test later.
- Do not trust failures from `build` before `build-nvidia` proves the test is sound.
- Keep `_Ctx` and non-`_Ctx` paths separate unless the module already unifies them cleanly.
- Reuse module-local helpers and fixture infrastructure before creating one-off patterns.
- Do not switch into performance work before correctness and coverage status are stable.

## Finish only when

- the focused test passes in `build-nvidia`
- the local implementation passes the same focused test in `build`
- coverage is refreshed when applicable
- the result states which NVIDIA NPP behavior was matched and what remains unverified
