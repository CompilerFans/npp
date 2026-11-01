# nppiResize* 5090 (NPP 13.4) semantics notes — package g4

Baseline: RTX 5090, NVIDIA NPP 13.4.  Every statement below was recovered from
the recorded goldens in `artifacts/5090/npp_records` (probe suites
`NppiResizeSqrPixelDiff.*`, `NppiResizeProbeDiff.*`,
`NppiResizeFamilyProbeDiff.*`, `NppiResizeBatchProbeDiff.*`) and from direct
NPP probes on the 5090.

## 1. Coordinate mapping (SqrPixel)

Documented formula, confirmed by the NN coordinate-decode cases:

```
adjF   = 1 / nFactor
adjS   = nShift * adjF + (1 - adjF) * 0.5
srcXf  = adjF * dstX - adjS        (dstX is the ABSOLUTE destination column,
                                    no oDstROI offset is applied)
srcYf  = adjF * dstY - adjS
```

Precision: `64f` uses double throughout; every other dtype evaluates one
rounded float32 FMA (`fmaf(af, dx - off, base)` with
`off = dstROI.x - 0.5`, `base = srcROI.x - 0.5` for nppiResize, and
`off = 0`, `base = -adjS` for SqrPixel).

`NPPI_INTER_NN` selects `nearest = floor(srcXf + 0.5)` (ties round up).

## 2. Clip rule (measured)

Destination pixels are written iff the sample coordinate lies in

```
oSrcROI.x - 0.5 * nXFactor <= srcX < oSrcROI.x + oSrcROI.width
```

(the lower bound is inclusive, the upper strict; same for y).  Pixels outside
are **not written** (destination poison stays).

`NPPI_INTER_SUPER` refuses every factor `>= 1` with `NPP_RESIZE_FACTOR_ERROR`
(including exactly 1.0), both for SqrPixel and for nppiResize.

The measured rectangle is not captured perfectly for shifted combinations:
`NppiResizeProbeDiff.Clip_8u_C1` still reports a handful of destination
pixels whose write decision differs at the corners (see section 8).

## 3. Validation order (nppiResizeSqrPixel, measured)

Order of the checks, each confirmed by a two-violation probe:

1. `nXFactor <= 0 || nYFactor <= 0` -> `NPP_RESIZE_FACTOR_ERROR`.
2. `pSrc == NULL || pDst == NULL` -> `NPP_NULL_POINTER_ERROR`.
3. source image size: negative -> `NPP_SIZE_ERROR`; `width == 0 || height == 0`
   -> `NPP_NO_ERROR` early exit; `< 2` -> `NPP_RECTANGLE_ERROR`.
4. destination ROI negative width/height -> `NPP_SIZE_ERROR`; zero width or
   height -> `NPP_NO_ERROR` early exit.
5. `nSrcStep` too small -> `NPP_STEP_ERROR`.
6. source ROI rectangle (`width <= 0` or `height <= 0`) -> `NPP_RECTANGLE_ERROR`.
7. `eInterpolation` not a known mode -> `NPP_INTERPOLATION_ERROR`.
8. `nDstStep` too small -> `NPP_STEP_ERROR`.
9. `NPPI_INTER_CUBIC2P_*` -> `NPP_INTERPOLATION_ERROR` (after the step
   checks: `sqr_code/*_roi_beyond` records `NPP_STEP_ERROR`).
10. source ROI has no intersection with the source image ->
    `NPP_WRONG_INTERSECTION_ROI_WARNING` (no pixels written).

## 4. Host helpers

Implemented in `src/nppi/nppi_geometry_transforms/nppi_resize_executor.h`
(see the header comment): `nppiGetResizeRect`, `nppiGetResizeTiledSourceOffset`
and `nppiResizeAdvancedGetBufferHostSize_8u_C1R`.

`nppiGetResizeRect` checks the source size before the null pointer (golden
`resize-host/getrect/error/null-over-size` -> `NPP_SIZE_ERROR`).

## 5. Interpolation kernels (impulse-response round)

Every kernel was recovered from `NppiResizeProbeDiff.Impulse_*` and
`NppiResizeSqrPixelDiff.WeightDecode_32f_C1`.

| mode | kernel (measured) |
|---|---|
| NPPI_INTER_NN | `floor(srcX + 0.5)` (ties up), single tap |
| NPPI_INTER_LINEAR | 2 taps `floor`, `floor+1`, weights `1-t`, `t` |
| NPPI_INTER_CUBIC | 4-point Lagrange stencil `{x0-1, x0, x0+1, x0+2}` |
| NPPI_INTER_LANCZOS | 6 taps `x0-2 .. x0+3`, weights from a 301-entry float32 table sampled at d = i/100 and interpolated linearly in double, then normalised by their sum |
| NPPI_INTER_SUPER | area average over the source interval `[af*(x-shift), af*(x-shift+1))` (width `1/f`), normalised by the box area |
| NPPI_INTER_CUBIC2P_* | rejected with `NPP_INTERPOLATION_ERROR` |

The Lanczos table is NPP's `constTblLanczos3` (`libnppig`); its values equal the
exact double `sinc(d)*sinc(d/3)` rounded to float32 (NPP's table differs from
that rounding by at most 1 ulp).  `nppi_resize_kernel.cu` embeds the 301
measured samples, matching every recorded 32f/64f impulse response bit-exactly.

Interpolation is **separable** (horizontal pass, then vertical), and the
rounding of that order is observable: a value of exactly 128.5 stays 128.  The
kernel therefore accumulates the x taps into a per-row sum first and then
combines the rows.

Taps outside the image are clamped to the edge.  Integer outputs round half
away from zero and saturate.

## 6. AC4 alpha

`nppiResizeSqrPixel_*_AC4R` and `nppiResize_*_AC4R` resample the three colour
channels and **leave the alpha channel untouched** (the poison fill stays in
the golden).  mcMPP replicates that: the AC4 templates process 3 channels of a
4-channel pixel.

## 7. nppiResize / nppiResizeBatch

* mapping: `srcX = oSrcROI.x + (dstX - oDstROI.x + 0.5) * (oSrcROI.width /
  oDstROI.width) - 0.5` (same for y).  The coordinate is evaluated in the
  pixel-centre form (`(dx - (dstROI.x - 0.5)) * af + (srcROI.x - 0.5)`) so
  that the float32 rounding matches NPP.
* clip: `ix0 - 0.5 <= srcX < ix1` where `[ix0, ix1)` is the intersection of
  `oSrcROI` with the image.
* interpolation taps are clamped to `[ix0, min(ix1 - 1 + radius, imageEnd-1)]`
  with `radius` = 0 (NN), 1 (linear), 2 (cubic) and 3 (Lanczos).  The left
  edge is the ROI start (measured with the coordinate-coded `cubic_*` and
  `lanczos_*` probes).
* error paths are much more lenient than SqrPixel: negative/zero source or
  destination sizes, degenerate ROIs and an empty ROI intersection all return
  `NPP_RESIZE_NO_OPERATION_ERROR`; null pointers and bad steps are checked
  first; `NPP_INTERPOLATION_ERROR` for unsupported modes;
  `NPP_RESIZE_FACTOR_ERROR` when `NPPI_INTER_SUPER` is asked for factor >= 1.
* `nppiResizeBatch` is the per-image `nppiResize` of the descriptor list, but
  it does **not** clamp taps to the per-image source ROI (measured with the
  `32f_C1/linear_roi` batch golden: the first destination column blends with
  the pixel left of the ROI).  `_Advanced` uses the same path.

## 8. Known deviations (mcMPP vs NPP 13.4)

The differential suite (`NppiResize*`, 41 tests) passes 32 tests.  The open
cases are:

* `nppiResize/16u_C1/lanczos_roi` and a few SqrPixel 16u/16s Lanczos values are
  off by one LSB at exact `.5` rounding boundaries.
* `nppiResizeSqrPixel_8u_C1R_Advanced` (LANCZOS3_ADVANCED): the recorded
  impulse response is not separable (a 6x6 pattern with a 2x2 core) and cannot
  be produced by a single-pass normalised Lanczos-3.  The internal two-pass
  buffer layout is not reproducible; mcMPP runs a single pass.
  (`NppiResizeProbeDiff.Impulse_Advanced_8u_C1` and
  `NppiResizeSqrPixelDiff.Advanced_8u_C1` remain open.)
* `NPPI_INTER_SUPER` with factor < 0.5 skips the first destination row/column
  in NPP; mcMPP writes it (`super_big_down4`).
* Shifted SqrPixel clip corners: NPP's write rectangle for ROI+shift
  combinations does not follow the `oSrcROI.x - 0.5*nXFactor` rule exactly
  (`NppiResizeProbeDiff.Clip_8u_C1`, `Ramp_32f_C1` x-y shift cases).
* nppiResize source ROIs that extend beyond the image still differ in the
  destination mask corner (`resize/nn_roi_beyond`,
  `resize_contract/src_roi_partial`, `nppiResize/8u_C1/roi_beyond`,
  `super_full_dstsub`).
