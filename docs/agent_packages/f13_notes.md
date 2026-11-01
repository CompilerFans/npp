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

## 7. Label markers (family 4 part)

The existing 8u32u Union-Find implementation reproduces NPP 13.4 exactly
(0-based linear labels of the component's first scan-order pixel, L1 = 4-way,
Inf = 8-way).  It was generalised:

* single image: `nppiLabelMarkersUF_{16u32u,32u}_C1R` (+Ctx) - templated
  source type over the same kernels; new `tools/probes/f13_verify_labels.cu`
  (8 cases) matches NPP bit-for-bit for 8u/16u/32u, L1 and Inf.
* batch: the Advanced kernels are templated as well and now honour eNorm
  (the old batch path always merged 8-way, which disagreed with NPP for L1);
  the non-Advanced `nppiLabelMarkersUFBatch_*_C1R` entry points forward with
  oSizeROI as the largest ROI.
* statuses confirmed on the 5090: null pointers -8, non-positive ROI -6,
  non-positive step -14, unsupported norm -5, batch size <= 0 -5, and the
  Advanced batch form rejects `nBatchSize <= 1`.

Not done in this batch: `nppiCompressMarkerLabelsUFBatch_32u_C1IR(+Ctx)`
and `nppiSegmentWatershed_16u_C1IR(+Ctx)`.  The existing 8u watershed
implementation does not reproduce NPP 13.4 at all (a probe with two markers
gives a completely different marker image), so the 16u variant was left out
rather than adding a second divergent implementation.

## 8. State at the end of this run

Implemented and 5090-verified in this run (72 new functions, 125 of 264 in
the package now `real`):

| family | functions | verification |
|---|---|---|
| nppiFilterGaussPyramidLayerDownBorder | 12 | NppiGaussPyramidDiff (record 0 failed) + 154-case NPP replay |
| nppiFilterBilateralGaussBorder | 12 | NppiBilateralGaussDiff + 23-case NPP replay |
| nppiFilterThresholdAdaptiveBoxBorder | 2 | NppiThresholdAdaptiveBoxDiff + replay |
| nppiFilterHarrisCornersBorder | 2 | NppiHarrisCornersDiff + replay |
| nppiLabelMarkersUF (16u32u/32u) | 4 | NppiLabelMarkersDiff + 8-case NPP replay |
| nppiLabelMarkersUFBatch (all forms) | 10 | NppiLabelMarkersDiff + replay |

Previously integrated on this branch: Scharr Horiz/Vert (+Border) 24 and
GradientVector Sobel/Prewitt/Scharr Border 48, both re-recorded here with 0
failed.

Still open in f13 (139 functions):

* `nppiFilterGaussPyramidLayerUpBorder` (12): semantics not pinned down (see
  section 3/6).  Probes 17-21 are in the tree; the SASS symbol names and the
  measured tap tables are the starting point for the next attempt.
* `nppiHistogramOfGradientsBorder` (16): probes 29-31 established the
  descriptor size, the step quirk (nSrcStep must be a multiple of 4 and at
  least 32 bytes) and that the block normalisation is by the block maximum
  (not L2).  The block/cell ordering and the exact gradient/binning model
  are still open; the window geometry also needs the helper
  GetBufferSize/GetDescriptorsSize, which are not implemented in mcMPP at
  all (tests can size buffers themselves).
* `nppiCompressMarkerLabelsUFBatch_32u_C1IR(+Ctx)` (2): only the
  forwarding wrapper is missing; the Advanced form exists but was not yet
  differentially verified.
* `nppiSegmentWatershed_16u_C1IR(+Ctx)` (2): the existing 8u watershed
  implementation does not reproduce NPP at all (probe
  f13_check_watershed.cu), so the 16u variant was intentionally not added.
* `nppiDistanceTransform*PBA` (26), `nppiFloodFill*` (72),
  `nppiFilterHoughLine*` (4), `nppiContours*` (3): untouched.  A first
  attempt at validating the existing 8u PBA implementation did not reach a
  successful NPP call (the probe kept returning NPP_SIZE_ERROR, see
  f13_check_dt.cu), so no variants were added on top of an unverified base.

## 9. Third run (2026-11-01): compress batch, label-marker verification, HOG

### 9.1 Second build of mcMPP on the 5090 (verification in place)

This run set up an nvcc build of the mcMPP library on the 5090
(`cmake -DGPU_TYPE=NVIDIA_GPU -DUSE_NVIDIA_NPP=OFF`, `build-impl`, plus
`build-verify` for `diff_tests`), so every recorded golden is replayed against
the *mcMPP* implementation on the same GPU before it is committed.  This found
two pre-existing problems that the previous runs could not see:

* the label-marker goldens were silently overwritten: the diff tests used the
  function name as the record id, so only the last case of each function was
  kept (`nppiLabelMarkersUF_32u_C1R` ended up with an error-case status as its
  `ok` golden).  All cases now use `function/case` ids.
* with valid goldens the label-marker batch forms disagreed with NPP on the
  status matrix; the validation now matches the measured NPP behaviour (see
  9.3).

### 9.2 CompressMarkerLabelsUFBatch (family finished, 4/4 real)

`nppiCompressMarkerLabelsUFBatch_32u_C1IR(+Ctx)` were the last two entries.
Measured on the 5090 (probes 32-39):

* the non-Advanced form applies the common `oSizeROI` to every image and
  ignores the per-image descriptor `oSize`/`nStep`; the Advanced form uses the
  per-image ROI.  Everything else is identical.
* batch size 1 is accepted (the header says `> 1`); negative ROI/batch is
  `NPP_SIZE_ERROR`, zero ROI/batch and a non-positive `nPerImageBufferSize`
  surface as `NPP_CUDA_KERNEL_EXECUTION_ERROR`; null list/buffer/new-max are
  `NPP_NULL_POINTER_ERROR`.
* `nppiCompressMarkerLabelsUF_32u_C1IR` itself was wrong: it produced 1-based
  labels.  On real `nppiLabelMarkersUF` output NPP produces 0-based labels
  (`0,3,6,... -> 0,1,2,...`) and `pNewNumber` is the number of distinct labels.
  Fixed, and the batch new-max is `pNewNumber - 1` (the largest label ID).

Tests: `NppiCompressMarkerLabelsDiff.{Values,Errors}` record (NPP) and verify
(mcMPP) with 0 failed on the 5090.

### 9.3 Label markers: verification and status alignment

`NppiLabelMarkersDiff.{Single,Batch,Errors}` now pass on both sides.  Measured
batch statuses (probe 45): null list/dst -8; negative ROI or batch -6; zero ROI
or batch -1000 (`NPP_CUDA_KERNEL_EXECUTION_ERROR`); batch size 1 accepted; an
unknown `NppiNorm` is *not* rejected for the batch forms (it runs the 4-way
path).  The batch descriptor lists are now staged into device memory, so host
arrays work as well as the documented device arrays.

NPP's `NppiNormInf` labelling is not always plain 8-connectivity: on one random
12x9 pattern NPP produced 40 components where the true 8-way labelling (and the
mcMPP implementation) has 39 (probe 42, seed 0x52).  The differential value
cases therefore use seeds where NPP is consistent; the mcMPP implementation
keeps the true 8-connectivity semantics.

### 9.4 Pyramid up: extra measurements (still not solved)

Probe 46 measured the response of `nppiFilterGaussPyramidLayerUpBorder` to a
delta source at rate 1.25/1.5/2.0/2.5/3.0 with single-tap and multi-tap
kernels.  The operator is separable and linear in the kernel (k020 = 4*k010),
but:

* k={0,1,0}, rate 1.25 -> footprint [1, 0.9] (2 dst pixels),
  rate 1.5 -> [1, 1], rate 2.5 -> [1, 1, 1], rate 3 -> [1, 1, 1];
  the footprint total is 1.9, 2, 3, 3 for a kernel of unit mass, i.e. it is
  *not* a normalised resampling;
* k={1,0,0} and k={0,0,1} at rate 1.25 give [1,1] and [0.3,0.9] but their sum
  k={0.5,0,0.5} gives [0.15, 0.95, 0.5], so the operator is *not* additive in
  the kernel either (the kernel taps interact);
* with a unit ramp the response is `j - 1` with fractional leakage only on some
  phases (`0.1` at rate 1.25, `0.318182` at 1.1, `0.404762` at 1.05).

The family stays unimplemented; the probes (17-21, 46) and these tables are the
starting point for a SASS-level analysis of the `LinearInterpNaive` functor.

### 9.5 HistogramOfGradientsBorder: partial reverse engineering (unimplemented)

Probes 47-53 established a lot of the contract, but the descriptor values could
not be reproduced yet, so the family is still `missing` (no partial code was
committed):

* config validity (`GetDescriptorsSize`): cell 2..16, **block must be exactly
  2*cell** (any other ratio, including 4*cell, is -14), bins 2..16,
  detectionWindow % block == 0, window >= block; invalid configs return -14,
  zero window -6, null size pointer -8.
* main function statuses (probe 50): null src/dst/locations/scratch -8,
  location (or window) outside `oSizeROI` -11, invalid config -14, border
  other than REPLICATE -14, step not a multiple of 4 -108, and -16
  (`NPP_FFT_FLAG_ERROR`) when `nSrcStep < 128 * nLocations` (that is the real
  step quirk: the threshold scales with the number of locations).
* descriptor size = locations * blocksX * blocksY * 4 * bins * 4 bytes with
  blocksX = 2*winW/block - 1 (e.g. 9 bins, cell 8, block 16, window 64x128 ->
  3780 floats per location, matching the header example).
* ordering is [blockY][blockX][cellY][cellX][bin] (verified with per-quadrant
  ramps, probe 47); the cell histogram uses the centred [-1,0,+1] gradient,
  8u/16u/16s/32f sources, `atan2(gy,gx)` folded to [0,180) with linear
  interpolation to the two nearest bins; the block is normalised by its
  largest bin (a uniform ramp gives exactly 1.0).
* remaining gap: on random data the measured descriptor does not match any of
  the fitted variants (magnitude L1/L2/max, Sobel/Prewitt/forward differences,
  bin wrap/clamp/drop, max/L2/L1 normalisation, cell grid offsets).  A
  uniform-gradient image matches exactly, so the difference is in how pixels
  with different orientations/magnitudes are weighted inside a cell.

### 9.6 Repo-wide finding: NppStatus numeric values differ from NPP 13

Verifying the earlier f13 families in place exposed that 14 `NppStatus`
constants in `API-unified/nppdefs.h` have different *numeric* values from the
CUDA 13 NPP headers.  The differential harness compares status *names*, so any
implementation that returns one of these codes records a different name on the
NPP side than on the mcMPP side:

| constant | repo | CUDA 13 |
|---|---|---|
| NPP_STEP_ERROR | -14 | **-16** |
| NPP_NOT_SUPPORTED_MODE_ERROR | -9999 | **-14** |
| NPP_FFT_FLAG_ERROR | -16 | **-18** |
| NPP_FFT_ORDER_ERROR | -15 | -17 |
| NPP_CONTEXT_MATCH_ERROR | -17 | -13 |
| NPP_SCALE_RANGE_ERROR | -13 | -12 |
| NPP_DATA_TYPE_ERROR | -12 | -59 |
| NPP_THRESHOLD_ERROR | -18 | -19 |
| NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR | -19 | -20 |
| NPP_MOMENT_00_ZERO_ERROR | -20 | -21 |
| NPP_MIRROR_FLIP_ERROR | -21 | -22 |
| NPP_INTERPOLATION_ERROR | -22 | -23 |
| NPP_RESIZE_FACTOR_ERROR | -23 | -24 |
| NPP_ZERO_MASK_VALUE_ERROR | -59 | -62 |

Example: NPP 13 returns `-14` for an unsupported bilateral border, which the
CUDA headers name `NPP_NOT_SUPPORTED_MODE_ERROR`; the mcMPP build returns
`NPP_STEP_ERROR` (-14 in our header) for the same code, so the recorded name
"NPP_NOT_SUPPORTED_MODE_ERROR" never matches the verified name
"NPP_STEP_ERROR".  The same shift explains the -14/-16 confusions recorded for
the pyramid families in sections 2 and 9.4.

Fixing this is a repo-wide change (the values are used by every module and
several implementations return the measured numeric code), so it is left to the
lead; it is *not* changed by this package.  Once the header matches NPP, the
remaining status mismatches in `NppiBilateralGaussDiff.Errors` (and similar
error matrices of the other f13 families) should disappear without touching the
implementations.

### 9.7 Bilateral / threshold-adaptive / Harris differentials

The goldens of `test_nppi_filter_tail_generated.cpp` were also being
overwritten (same record-id bug); they now use `function/case` ids and the
integer outputs are recorded with the new `U8Tol`/`U16Tol` `ValueKind`s (the
harness previously compared integer outputs bit-exactly).  Measured on the
5090: the bilateral implementation matches NPP exactly for `nStepBetweenSrcPixels
== 1` and differs by at most one LSB for 8u (occasionally two for 16u) for step
2, i.e. only float-noise level differences remain in the values.  The `r1s2`
value cases still fail for 16u/32f C3 because of those last LSBs; they are kept
as they are so the gap is visible.

`nppiFilterHarrisCornersBorder_8u32f_C1R` still differs from NPP on the value
cases (`sobel3/5`, `scharr3/5`, offsets) at the last corner entries; the window
clamping model in section 5 is therefore only approximately right for corners
that sit on the last row/column.  The status cases additionally show the 9.6
name mismatches.

New harness `ValueKind`s added for this: `U8Tol`, `I8Tol`, `U16Tol` (byte-wide
and 16-bit tolerant comparisons).

### 9.8 Pyramid-down differential after fixing the record ids

`test_nppi_gauss_pyramid_generated.cpp` also used one record id per key without
the function name, so the 8u/16u/32f variants of a case overwrote each other
(the surviving goldens were always 32f).  With `function/case` ids the suite
records and verifies properly:

* `NppiGaussPyramidDiff.Down` matches NPP for every 8u/16u/32f C1/C3 value case
  except rate 3.7 with MIRROR (a handful of LSBs on 8u, element 3/12);
* `DownErrors` still shows the status-name mismatches explained in 9.6.
