# Rounding Mode Impact on Super Sampling Error

## Question

**Does the observed error in fractional scale super sampling relate to specific rounding modes?**

## Background

In the fractional scale test (`test_fractional_scale.cpp`), we observed:
- Expected value: 16.333
- NPP result: 16
- Difference: 0.333

This raises the question: does the rounding mode affect this error?

## Rounding Modes in NPP

NPP supports three main rounding modes (from `nppdefs.h`):

### 1. NPP_RND_NEAR (Nearest-Even)
Round to nearest integer, ties to even:
- 0.5 → 0 (even)
- 1.5 → 2 (even)
- 2.5 → 2 (even)
- 3.5 → 4 (even)
- 127.5 → 128 (even)
- 128.5 → 128 (even)

**Also called**: `NPP_ROUND_NEAREST_TIES_TO_EVEN` (IEEE 754 standard)

### 2. NPP_RND_FINANCIAL (Nearest-Away)
Round to nearest integer, ties away from zero:
- 0.5 → 1
- 1.5 → 2
- 2.5 → 3
- 3.5 → 4
- 127.5 → 128
- 128.5 → 129

**Also called**: `NPP_ROUND_NEAREST_TIES_AWAY_FROM_ZERO`

### 3. NPP_RND_ZERO (Toward-Zero)
Truncate toward zero:
- 0.5 → 0
- 1.5 → 1
- 2.5 → 2
- 3.5 → 3
- 127.5 → 127
- 128.5 → 128

**Also called**: `NPP_ROUND_TOWARD_ZERO`

## Important Finding

### nppiResize Does NOT Accept Rounding Mode Parameter

```cpp
NppStatus nppiResize_8u_C1R(
    const Npp8u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
    Npp8u * pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
    int eInterpolation  // Only interpolation mode, NO rounding mode
);
```

**Rounding mode is NOT configurable for nppiResize.**

NPP likely uses **Nearest-Even** (IEEE 754 standard) internally, which is the most common default.

## Error Analysis by Case

### Case 1: Exact Integer Result

**Example**: Raw value = 20.0

| Rounding Mode | Result | Error |
|---------------|--------|-------|
| Nearest-Even  | 20     | 0.0   |
| Nearest-Away  | 20     | 0.0   |
| Toward-Zero   | 20     | 0.0   |

**Conclusion**: No impact when result is exact integer.

### Case 2: Non-0.5 Fractional Result

**Example**: Raw value = 16.333

| Rounding Mode | Result | Error |
|---------------|--------|-------|
| Nearest-Even  | 16     | 0.333 |
| Nearest-Away  | 16     | 0.333 |
| Toward-Zero   | 16     | 0.333 |
| Toward-Inf    | 17     | 0.667 |

**Conclusion**: All "nearest" modes produce **identical results** and **identical errors**.

**NPP Result**: 16 ✓ (matches Nearest-Even/Away/Zero)

### Case 3: Exact 0.5 Fractional (Even Target)

**Example**: Raw value = 127.5 (target 128 is even)

| Rounding Mode | Result | Error | Note |
|---------------|--------|-------|------|
| Nearest-Even  | 128    | 0.5   | Round to even |
| Nearest-Away  | 128    | 0.5   | Round away from 0 |
| Toward-Zero   | 127    | 0.5   | Truncate |
| Toward-Inf    | 128    | 0.5   | Ceiling |

**Conclusion**: All modes have **same absolute error (0.5)**, but different directions.

### Case 4: Exact 0.5 Fractional (Odd Target)

**Example**: Raw value = 128.5 (target 128 is even, 129 is odd)

| Rounding Mode | Result | Error | Note |
|---------------|--------|-------|------|
| Nearest-Even  | 128    | -0.5  | Round to even |
| Nearest-Away  | 129    | +0.5  | Round away from 0 |
| Toward-Zero   | 128    | -0.5  | Truncate |
| Toward-Inf    | 129    | +0.5  | Ceiling |

**Conclusion**: Nearest-Even differs from Nearest-Away only for **odd→even** ties.

## Detailed Comparison Table

| Value | Nearest-Even | Nearest-Away | Toward-Zero | Toward-Inf |
|-------|--------------|--------------|-------------|------------|
| 0.4   | 0            | 0            | 0           | 1          |
| 0.5   | 0 ★          | 1            | 0           | 1          |
| 0.6   | 1            | 1            | 0           | 1          |
| 1.4   | 1            | 1            | 1           | 2          |
| 1.5   | 2            | 2            | 1           | 2          |
| 2.4   | 2            | 2            | 2           | 3          |
| 2.5   | 2 ★          | 3            | 2           | 3          |
| 3.5   | 4            | 4            | 3           | 4          |
| 15.5  | 16           | 16           | 15          | 16         |
| 16.5  | 16 ★         | 17           | 16          | 17         |
| 127.5 | 128          | 128          | 127         | 128        |
| 128.5 | 128 ★        | 129          | 128         | 129        |

★ = Nearest-Even differs from Nearest-Away (tie-breaking to even)

## Answer to Original Question

### Q: Is the 0.333 error related to rounding mode?

**A: Yes and No.**

**YES**: The error exists because we must round a fractional result (16.333) to an integer.

**NO**: The rounding mode does NOT affect the error magnitude for non-0.5 cases.
- All "nearest" rounding modes produce **identical results** (16) for 16.333
- Error magnitude is **0.333** regardless of Nearest-Even vs Nearest-Away
- Rounding mode only matters for **exact 0.5 ties**

### Verification

For the observed test case:
```
Raw value: 16.333
Nearest-Even: 16 (error = 0.333)
Nearest-Away: 16 (error = 0.333)
Toward-Zero:  16 (error = 0.333)

NPP result: 16 ✓
Error: 0.333 ✓
```

**All nearest rounding modes produce the same result.**

## Maximum Possible Error

For "nearest" rounding modes (Even or Away):
- **Maximum error: 0.5 pixels**
- This is fundamental to integer quantization
- Cannot be reduced by changing rounding mode

For other modes:
- Toward-Zero: max error ≈ 1.0 (always underestimates)
- Toward-Inf: max error ≈ 1.0 (always overestimates)

## NPP's Likely Rounding Mode

Based on:
1. No rounding mode parameter in nppiResize API
2. Error magnitude consistent with nearest rounding
3. IEEE 754 standard compliance (CUDA standard)

**NPP likely uses Nearest-Even (NPP_RND_NEAR) internally.**

However, since the difference only appears for exact 0.5 ties, and most super sampling results are not exactly X.5, the choice between Nearest-Even and Nearest-Away has **minimal practical impact**.

## Implications for Testing

### 1. Expected Error Range

When comparing NPP results to reference implementation:
```cpp
float expected = calculateExpectedValue();
int nppResult = getNppResult();

// For nearest rounding:
EXPECT_NEAR(nppResult, expected, 0.5);  // Maximum possible error

// More strict for non-tie cases:
if (fabs(expected - round(expected)) > 0.01) {  // Not a tie
  EXPECT_NEAR(nppResult, round(expected), 0.49);
}
```

### 2. Handling Exact Ties

For exact 0.5 cases, both results may be valid:
```cpp
float expected = 128.5;
int nppResult = getNppResult();

// Accept either 128 or 129
EXPECT_TRUE(nppResult == 128 || nppResult == 129);
```

### 3. Tolerance Setting

For test assertions:
```cpp
// Observed in tests:
// - Exact cases: error = 0
// - Non-0.5 fractional: error ≤ 0.5
// - Typical error: 0.1 - 0.4

// Conservative tolerance: ±1 (covers rounding + minor FP error)
EXPECT_NEAR(result, expected, 1);

// Strict tolerance: ±0.5 (pure rounding error)
EXPECT_NEAR(result, expected, 0.5);
```

## Summary

### Key Findings

✅ **Rounding mode is NOT configurable** for nppiResize

✅ **NPP likely uses Nearest-Even** (IEEE 754 standard)

✅ **Error magnitude is the same** for Nearest-Even vs Nearest-Away (for non-0.5 cases)

✅ **Maximum error is 0.5** (fundamental to nearest rounding)

✅ **Observed 0.333 error is normal** rounding error, not a bug

### Recommendations

1. **Testing**: Use tolerance of ±0.5 for super sampling tests
2. **Exact 0.5 ties**: Accept either rounded value
3. **Error analysis**: Focus on error magnitude, not direction
4. **Implementation**: Use nearest rounding (std::round or roundf) in reference code

## Test Code

See `test_rounding_mode.cpp` for complete analysis and verification.

### Compilation
```bash
g++ -o test_rounding_mode test_rounding_mode.cpp -std=c++11
```

### Execution
```bash
./test_rounding_mode
```

## References

- NPP rounding modes: `/usr/local/cuda/include/nppdefs.h`
- IEEE 754 standard: Nearest-Even is the default rounding mode
- Test results: `test_fractional_scale.cpp` output
