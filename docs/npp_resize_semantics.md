# nppiResize* 5090 (NPP 13.4) semantics notes — package g4

Baseline: RTX 5090, NVIDIA NPP 13.4.  Every statement below was recovered from
the recorded goldens in `artifacts/5090/npp_records` (probe suites
`NppiResizeSqrPixelDiff.*`) by replaying the deterministic inputs of
`test/unit/diff/test_nppi_resize_sqr_pixel_generated.cpp` in a Python model.

## 1. Coordinate mapping (SqrPixel)

Documented formula, confirmed by the NN coordinate-decode cases:

```
adjF   = 1 / nFactor
adjS   = nShift * adjF + (1 - adjF) * 0.5
srcXf  = adjF * dstX - adjS        (dstX is the ABSOLUTE destination column,
                                    no oDstROI offset is applied)
srcYf  = adjF * dstY - adjS
```

Precision (measured against the 32f big goldens, max |err| of the replay):

| dtype | adjF/adjS | coordinate arithmetic                    | replay error |
|-------|-----------|-------------------------------------------|--------------|
| 64f   | double    | double                                    | 7e-8         |
| others| float32   | one rounded float32 FMA (`fmaf(f, d, -s)`)| 1.2e-7       |
| others| float32   | separate mul/sub in float32               | 2.8e-5       |

The last row fails the harness tolerance, so the float32 path must use a single
rounding (CUDA `fmaf`) or the equivalent double-then-cast.  `nYFactor` enters
the same way (`adjF = 1/nYFactor`).

`NPPI_INTER_NN` selects `nearest = floor(srcXf + 0.5)` (ties round up: a
coordinate of exactly 4.5 samples pixel 5).  This is true for 16u/32f/64f.

`NPPI_INTER_LINEAR` blends pixels `floor(srcXf)` and `floor(srcXf)+1` with
weight `frac = srcXf - floor(srcXf)`; integers round the result with
`floor(v + 0.5)` and saturate to the dtype range (16u verified exactly,
0 mismatches over 54 written pixels).  64f matches the double model to 7e-8.

## 2. Clip rule (measured, still being probed)

Two different rules were observed; a dedicated probe round
(`NppiResizeProbeDiff.*`) pins them down:

* NN: a destination pixel is written iff
  `oSrcROI.x - 1 < srcXf < oSrcROI.x + oSrcROI.width` (same for y).
  The strict `-1` lower edge reproduces every NN golden: `nn_srcroi_decode`
  writes a destination pixel whose sampled source pixel is `oSrcROI.x - 1`.
* LINEAR: written iff the sample position stays inside the ROI
  (`oSrcROI.x <= srcXf < oSrcROI.x + width`); `nonzero_shift` (f=1.5,
  shift 2/-1) clips the columns whose `srcXf` is negative and keeps the rest.

Pixels outside the clip region are **not written** (destination poison stays).

## 3. Validation order (nppiResizeSqrPixel, measured)

Order of the checks, each confirmed by a two-violation probe:

1. `nXFactor <= 0 || nYFactor <= 0` -> `NPP_RESIZE_FACTOR_ERROR`
   (also for `NPPI_INTER_SUPER` upscaling, i.e. factor > 1).
2. `pSrc == NULL || pDst == NULL` -> `NPP_NULL_POINTER_ERROR`.
3. source image size: negative -> `NPP_SIZE_ERROR`; `width == 0 || height == 0`
   -> `NPP_NO_ERROR` early exit; `< 2` -> `NPP_RECTANGLE_ERROR`.
4. destination ROI negative width/height -> `NPP_SIZE_ERROR`; zero width or
   height -> `NPP_NO_ERROR` early exit.
5. `nSrcStep` too small -> `NPP_STEP_ERROR`.
6. source ROI rectangle (`width <= 0` or `height <= 0`) -> `NPP_RECTANGLE_ERROR`.
7. `eInterpolation` not in the supported set -> `NPP_INTERPOLATION_ERROR`.
8. `nDstStep` too small -> `NPP_STEP_ERROR`.
9. source ROI has no intersection with the source image ->
   `NPP_WRONG_INTERSECTION_ROI_WARNING` (no pixels written).

The exact positions of steps 5–8 are constrained by the recorded pairs
(`order_step_and_dstneg` -> STEP, `order_dstneg_and_interp` -> SIZE,
`order_stepdst_and_interp` -> INTERPOLATION, `order_step_and_sroizerow` -> STEP).

## 4. Host helpers

Already implemented in `src/nppi/nppi_geometry_transforms/nppi_resize_executor.h`
(see the header comment): `nppiGetResizeRect`, `nppiGetResizeTiledSourceOffset`
and `nppiResizeAdvancedGetBufferHostSize_8u_C1R`.

## 5. Open items

* CUBIC / CUBIC2P_* / LANCZOS / SUPER kernels: `32f` goldens show a 4-tap
  separable kernel for CUBIC, but the recovered weights do not match
  Catmull-Rom at every phase (they do at frac = 0.5).  An impulse-response
  probe round is recorded to recover the exact taps.
* LANCZOS3_ADVANCED scratch usage / two-pass rounding.
* nppiResize / nppiResizeBatch ROI mapping and per-image descriptors.
