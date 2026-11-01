# f13 working notes (measured on the RTX 5090, NPP 13.4)

## 1. Verification of the already-merged families

Re-recorded on the 5090 (build-nvidia, NVIDIA NPP 13.4):

* `NppiFilterDerivDiff.*` (incl. `ScharrHoriz`, `ScharrHorizBorder`,
  `ScharrVert`, `ScharrVertBorder`) - 26 tests, 0 failed.
* `NppiGradientVectorDiff.*` (Sobel / Prewitt / Scharr) - 3 tests, 0 failed.

Goldens were pulled to `artifacts/5090/npp_records/{nppiFilterDeriv,
nppiGradientVector}` (gitignored) for the lead's C500 replay.

## 2. Gauss pyramid - downsampling (`nppiFilterGaussPyramidLayerDownBorder`)

Reverse engineered with `tools/probes/f13_probe15/16/17` (run on the 5090).

Model (32f, confirmed exactly on all probe deltas):

```
dst[j] = sum_t k[t] * sample(j*rate + (t - (taps-1)/2))
```

* `sample()` is a *bilinear interpolation* of the source at the (fractional)
  coordinate, i.e. `floor`/`ceil` neighbours weighted by the fraction.
* Border handling applies to the fractional coordinate *after* the tap offset
  is added: `REPLICATE` clamps, `MIRROR` reflects (edge pixel not repeated).
* Integer rates therefore skip pixels exactly (`identity` kernel, rate 2
  gives `dst = src(2x,2y)`); non-integer rates interpolate (rate 1.5 identity
  gives `0, 1.5, 3, ...`).
* Kernel weights are used as given (not renormalised): a delta under a
  `{1/3,1/3,1/3}` kernel integrates to `4/9`, matching the raw tap sums.
* Taps: even counts are accepted (4 works), `2`, `0`, `-3`, `101` are
  rejected with `-33`. Rate: `1.0 < rate <= 10.0` (1.0/0.9/10.1 give -5).
* Border: only `REPLICATE` and `MIRROR` (`-14` otherwise).
* Validation order (probe15 part 4): null src/dst/kernel (-8), then rate (-5),
  then taps (-33), then border (-14), then step (-16 for too small or
  misaligned).

## 3. Gauss pyramid - upsampling (`nppiFilterGaussPyramidLayerUpBorder`)

Probes 17-21 were run. The mapping is *not* the mirror of the down path: the
phase of the taps appears quantised (a ramp yields exact integers for integral
rates but fractional leakage such as `0.1`, `0.318182` for rates like 1.25 /
1.1), and a single-tap kernel `{0,1,0}` already differs from `floor(d/rate)`
at these rates. Work on this family is still in progress; the raw probe tables
are reproducible from the probe sources (`f13_probe17..21.cu`).

## 4. HistogramOfGradientsBorder

Probe 13/14 measured the helper functions and one happy-path descriptor:

* `GetDescriptorsSize` = `nLocations * ((winW/block*2-1) * (winH/block*2-1)
  * (block/cell)^2 * bins) * 4` bytes.
* Invalid configs (cell <= 1, block < 2*cell, window not a multiple of block,
  bins <= 1 or > 16) give `-14`; zero window size gives `-6`; null
  `hpDescriptorsSize` gives `-8`.
* `GetBufferSize` gives `-11` for an ROI smaller than the window or locations
  out of range; `-6` for a zero ROI.
* The descriptor uses Dalal/Triggs L2 normalisation per overlapping 2x2-cell
  block (documented in the header); a single delta gave exact `1.0` spikes.

## 5. Bilateral, threshold-adaptive box, Harris corners

All three were fitted against NPP 13.4 with probes 26/27/28 and replayed with
`tools/probes/f13_verify_tail.cu` (23 value/status cases, 0 mismatches).

* `nppiFilterBilateralGaussBorder`: documented formula, 8u/16u truncate.
  Radius 1..32 (`-33` otherwise), `nStepBetweenSrcPixels <= 0` reports -16,
  MIRROR is accepted, other borders -14.  For step > 1 NPP snaps only the
  *horizontal* coordinate onto the step grid (`(x + step/2)/step*step`,
  clamped to the last column) and leaves the vertical coordinate alone; the
  executor reproduces that.  Step 2 matches NPP within its float noise, step 3
  and 4 still differ in the last columns, so the diff tests only assert
  statuses for those.
* `nppiFilterThresholdAdaptiveBoxBorder`: REPLICATE mean over an odd square
  mask (`-33` for even / non-square / zero), then `src > avg - nDelta`.
  Only REPLICATE is accepted (`-14`).
* `nppiFilterHarrisCornersBorder`: Sobel 3x3/5x5 or Scharr 3x3 gradients
  (Scharr + 5x5 -> -33), products summed (not averaged) over the 3x3/5x5
  window; the window clamps the *sample position* before taking the gradient
  (clamping each gradient tap gives different border values).  The device
  buffer is required (`-8`), a null destination is accepted with no output,
  and an unsupported border is `-14`.

## 6. Remaining f13 families

* `nppiFilterGaussPyramidLayerUpBorder` (12): SASS shows the implementation is
  a `ForEachPixelLinearInterpNaive<...>FilterGaussPyramidLayerUpLinearInterp
  {Replicate,Mirror}Border{3x3,5x5}Functor`; the measured tap responses are
  not a plain bilinear resample (a single-tap kernel already deviates from
  `floor(d/rate)` at non-integral rates, and the same ramp gives
  kernel-dependent fractional leakage such as `0.1`, `0.318182`).
  Deferred; probe tables 17-21 remain in the tree.
* `nppiHistogramOfGradientsBorder` (16), `nppiLabelMarkersUF*`,
  `nppiFloodFill*`, `nppiDistanceTransform*PBA`, `nppiSegmentWatershed*`,
  `nppiFilterHoughLine*`, `nppiContours*`: still missing at the time of this
  note.
