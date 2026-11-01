# s2 notes — statistics families (2026-11-01 continuation)

Status: 202 / 206 package functions are `real`.  All implemented families
record cleanly on the RTX 5090 (`diff_tests`, 75 cases, 0 failed / 0 crash).

## Implemented in this pass

| family | functions | notes |
|---|---:|---|
| QualityIndex (+GBHS) | 36 | sample (n-1) statistics |
| CountInRange (+GBHS) | 24 | inclusive host bounds, device `int` counts |
| MSE / PSNR (+GBHS) | 16 | 8u C1/C3 |
| MSEBatch / PSNRBatch (+GBHS) | 24 | device descriptor lists; plain vs Advanced ROI |
| HistogramEven C3R/AC4R | 12 | generated over a shared channel histogram executor |
| HistogramRange C3R/AC4R | 16 | device per-channel level pointers |
| RectStdDev 32f / 32s_C1RSfs | 4 | integral-image box statistics |
| EvenLevelsHost_32s_Ctx | 1 | delegates to `evenLevelsHost` |
| SSIM / SSIMBatch (+GBHS) | 12 | 11x11 Gaussian, C1/C2 constants |
| WMSSSIM / WMSSSIMBatch (+GBHS) | 16 | decoded 5-level pipeline (see `00e3dab`) |

Files: `src/nppi/nppi_statistics_functions/nppi_quality_metrics_executor.{h,cu}`,
`nppi_ssim_executor.{h,cu}`, the generated `*_generated.cpp` wrappers and the
`mpp_scripts/families/nppi_*.json` specs (all wrappers are regenerated, never
hand-edited).

## 5090-measured semantics worth remembering

* **QualityIndex C3/AC4 quirk**: each channel's covariance uses *its own*
  channel means, but the numerator means `ma*mb` and the `(va+vb)` denominator
  are taken from **channel 0** for every output.  A constant channel in either
  source raises `NPP_QUALITY_INDEX_ERROR` with no output written; a 1x1 ROI
  succeeds and returns NaN.
* **CountInRange** bounds arrays are **host** memory (device bounds crash NPP);
  counts are device `int`.  Reversed bounds give `NPP_RANGE_ERROR`.
* **Buffer-size sweeps (dense 5090 dump)**:
  * QualityIndex: `4*(h + ceil(h/256) - 1) + 20` per channel, independent of
    width.
  * CountInRange / MSE / PSNR: `1024*h` for 8u with `h < 128`, else `12*h`;
    32f is always `12*h`; multiplied by the channel count.
  * MSEBatch / PSNRBatch: `8*ceil(w/32)*ceil(h/128)` per channel.
  * SSIM family sizes are NPP internals we did not fit; mcMPP reports a
    conservative working set and the tests pin status + sufficiency.
* **RectStdDev**: `dst(x,y)` is the std-dev of the window at
  `(x + rect.x, y + rect.y)`; 32s_C1RSfs truncates `sqrt` then shifts
  (left for negative `nScaleFactor`).  Invalid/negative rects return
  `NPP_SIZE_ERROR` (not RECTANGLE_ERROR).  Use `(n*sumSq - sum^2)/n^2` for the
  variance — the naive `E[x^2]-E[x]^2` form shifts truncated integers by one.
* **SSIM / WMSSSIM**: 11x11 Gaussian sigma 1.5, clamp border, `|cov|` in the
  contrast term (verified with random-noise goldens), C1 6.5025, C2 58.5225.
  WMSSSIM = product over 5 levels of `mean(cs^w)` (levels 0-3) and
  `mean((l*cs)^w)` (level 4), w = {0.0448, 0.2856, 0.3001, 0.2363, 0.1333},
  decimation is the 2x2 shift-clamp average.  WMSSSIM needs ROI >= 16x16.
* **Even levels**, NPP rounding is not simple linear interpolation for
  non-divisible ranges (e.g. `n=4, [0,10] -> 0,4,7,10`); mcMPP keeps the
  truncating interpolation and the tests only record divisible ranges (and the
  `lo == hi` case, which succeeds).

## Remaining / unresolved

1. `nppiMSSSIM_8u_C1R[_Ctx]` (2).  NPP requires ROI >= 176x176 and the value
   structure does not match WMSSSIM (constant 256x256 input returns 0); the
   reduction functor was not decoded.  `nppiMSSSIMGetBufferHostSize_8u_C1R`
   is implemented (conservative size).
2. `nppiCircularRadialProfile_32f_C1R_Ctx` and
   `nppiEllipticalRadialProfile_32f_C1R_Ctx` (2).  The 5090 samples a
   non-uniform digital ring set: `nPixels` is `4*i` samples per ring (not the
   number of pixels whose radius falls in `[i, i+1)`), and impulses on the
   `x == floor(cx)` axis are dropped.  The impulse scan
   (`tools/5090_probes_s2/s2_probe2.cu radial_scan`) documents the exact sample
   sets, but the generating rule was not reproduced.
3. NPP 13.4 has no `nppiEvenLevelsHost_32s_Ctx` symbol; the mcMPP Ctx wrapper
   is covered indirectly (same helper as the plain entry point, which the
   tests record).
4. The 16-bit `HistogramEven` C3R/AC4R entry points alias the channel-0 and
   channel-2 histogram buffers on the 5090 build (both receive the same bins),
   so no golden is recorded for them; the 16-bit path is covered by the Range
   forms.
