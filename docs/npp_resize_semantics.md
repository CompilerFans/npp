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

## 5. Interpolation kernels (impulse-response round)

Every kernel was recovered from `NppiResizeProbeDiff.Impulse_*`, which puts a
single impulse into a 32x32 `32f` image and records the whole destination.

| mode | kernel (measured) |
|---|---|
| NPPI_INTER_NN | `floor(srcX + 0.5)` (ties up), single tap |
| NPPI_INTER_LINEAR | 2 taps `floor`, `floor+1`, weights `1-t`, `t` |
| NPPI_INTER_CUBIC | **4-point Lagrange** stencil `{x0-1, x0, x0+1, x0+2}`: `-t(t-1)(t-2)/6, (t+1)(t-1)(t-2)/2, -t(t+1)(t-2)/2, t(t+1)(t-1)/6` (not Catmull-Rom; at t=0.5 it collapses to [-1/16, 9/16, 9/16, -1/16]) |
| NPPI_INTER_LANCZOS | 6 taps `x0-2 .. x0+3`, weights `sinc(d)*sinc(d/3)` **normalised by their sum** (verified to 1e-7 on the 32f big goldens) |
| NPPI_INTER_SUPER | area average: a source-space box of width `1/f` starting at `srcX-0.5`, weights = overlap/(width_x*width_y); only valid for downscaling |
| NPPI_INTER_CUBIC2P_* | rejected with `NPP_INTERPOLATION_ERROR` for every dtype |

Taps outside the image are clamped to the edge (measured on the cubic ramp
probe: the edge value equals the clamped-weight sum).  Integer outputs round
half away from zero (`16s` negative halves pin this down) and saturate.

`NPPI_INTER_SUPER` box placement is still only partially understood: the
`[srcX-0.5, srcX-0.5+1/f]` box matches f=0.5 and f=0.75 exactly but the
f=0.25 response is one destination pixel to the right of the predicted box
(`super/down4_phase0`).  The implementation uses the measured f=0.5/f=0.75
placement.

## 6. AC4 alpha

`nppiResizeSqrPixel_*_AC4R` and `nppiResize_*_AC4R` resample the three colour
channels and **leave the alpha channel untouched** (the poison fill stays in
the golden).  mcMPP replicates that: the AC4 templates process 3 channels of a
4-channel pixel.

## 7. nppiResize / nppiResizeBatch

* mapping: `srcX = oSrcROI.x + (dstX - oDstROI.x + 0.5) * (oSrcROI.width /
  oDstROI.width) - 0.5` (same for y); NN rounds half up.  Verified with
  coordinate-coded 32f goldens (`NppiResizeFamilyProbeDiff.Mapping_32f_C1`).
* clip: the source coordinate must satisfy `ix0 - 0.5 < srcX < ix1` where
  `[ix0, ix1)` is the intersection of `oSrcROI` with the image; taps clamp to
  the image edge.
* error paths are much more lenient than SqrPixel: negative/zero source or
  destination sizes, degenerate ROIs and an empty ROI intersection all return
  `NPP_RESIZE_NO_OPERATION_ERROR`; null pointers and bad steps are checked
  first; `NPP_INTERPOLATION_ERROR` for unsupported modes;
  `NPP_RESIZE_FACTOR_ERROR` when `NPPI_INTER_SUPER` is asked to upscale.
* `nppiResizeBatch` is exactly the per-image `nppiResize` of the descriptor
  list (verified for the basic variant with a 3-image batch).  The
  `_Advanced` descriptor/ROI variant is implemented the same way, but the
  recorded probe (`NppiResizeBatchProbeDiff.Advanced_8u_C1`) shows NVIDIA
  sampling outside the per-image ROI in a way we could not reproduce; treat
  that path as unverified.

## 8. Known deviations (mcMPP vs NPP 13.4)

* `NPPI_INTER_LANCZOS` weights differ by <= 5e-5 relative at small phases
  (the normalisation constant cannot be reproduced exactly; every other mode
  matches bit-exactly for the integer dtypes).
* `nppiResizeSqrPixel_8u_C1R_Advanced` (LANCZOS3_ADVANCED): NVIDIA's recorded
  impulse response is not separable (a 6x6 pattern with a 2x2 core) and does
  not depend on any shift parameter (the Advanced API has none), so its
  internal two-pass layout is not reproducible.  mcMPP runs a single-pass
  normalised Lanczos-3 instead.
* SqrPixel sub-ROI + shift + downscale combinations: the clip lower bound was
  fitted as `oSrcROI.x - 0.5 * nXFactor` (11/12 measured NN cases); one
  f=2/shift=0.3 case clips one destination column earlier in NVIDIA.
* `nppiResizeBatch_*_Advanced`: see section 7.
