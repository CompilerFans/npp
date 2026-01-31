# nppi_color_conversion.h development plan

Date: 2026-01-31

Assumptions:
- Missing APIs are defined as implemented != "yes" in coverage.csv (none found currently).
- Usage heat derived from repo scan (test/examples/docs/status/tools/ref_code + top-level md/txt).

## Priority strategy
1) Add tests for high-usage, untested APIs first (functional + precision).
2) Add tests for medium-usage, untested APIs next, using parameterized gtest to cover types/sizes.
3) Low/zero usage APIs last; batch by shared kernels/paths to maximize reuse.
4) When new missing APIs appear, implement in descending usage order and reuse shared color-conversion primitives.

## High-value test targets (untested but used)
None found in repo scan scope. Consider expanding scan scope if needed.

## Reuse guidance
- Consolidate common color space conversions into shared helpers to avoid per-API duplication.
- Prefer parameterized tests that share input generation and CPU reference paths across API variants.
