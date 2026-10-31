# f12 continuation notes (lead-derived from the recorded goldens)

The 5090 goldens pin these validation results; implement them in
nppi_filter_executor.{h,cu}.  Status names are compared by name.

## nppiFilter (2D, non-border; filterDiv / filter64f / filter32f)
- order: null -> negative ROI -> anchor -> dst step size -> empty ROI no-op
  -> divisor == 0.  Evidence: `step_small_vs_div0` -> STEP (step before
  divisor); `div0_vs_empty` -> NO_ERROR (empty before divisor).
- 64f (`nppiFilter_64f_C1R`, filter64f) does NOT validate the anchor:
  anchor {4,1} with a 3x3 kernel -> NO_ERROR.  The integer/32f variants do
  (anchor_high -> OUT_OFF_RANGE) and negative anchors still error.

## nppiFilterRow / nppiFilterColumn (integer, non-border)
- anchor > mask-1 is ACCEPTED (anchor 4, mask 3 -> NO_ERROR); very negative
  anchors (< 1 - ROI dim) still -> OUT_OFF_RANGE.
- step size is checked before divisor (`step_small_vs_div0` -> STEP) and the
  empty ROI no-op before divisor (`div0_vs_empty` -> NO_ERROR).
- a zero ROI in the CROSS axis is a SIZE error: row filter height 0 -> SIZE;
  column filter width 0 -> SIZE.  A zero ROI along the filter axis is the
  usual NO_ERROR no-op.

## nppiFilterRowBorder / nppiFilterColumnBorder
- source-offset validation runs BEFORE the border-mode check
  (`border_beats_offset` -> OUT_OFF_RANGE, not NOT_SUPPORTED_MODE).
- step size is checked before divisor; empty ROI no-op before divisor.

## nppiSumWindowRow / Column (non-border)
- mask <= 0 -> MASK_SIZE_ERROR (mask 0 and negative masks are errors; this
  replaces the zero-fill path for this family).
- anchor >= mask -> ANCHOR_ERROR.
- anchor <= 1 - ROI dim (the test's `aext` is exactly 1-W) -> OUT_OFF_RANGE.
- zero ROI in the CROSS axis -> SIZE_ERROR (row: height 0; column: width 0).
  Zero along the filter axis -> NO_ERROR no-op.
- value semantics: the window is the clipped running sum; a window sum is
  into an Npp32f destination (no divisor).

## nppiSumWindowRowBorder / ColumnBorder
- mask <= 0 -> MASK_SIZE_ERROR; anchor >= mask -> ANCHOR_ERROR; the rest of
  the border semantics stay as implemented.

## Remaining value failures to investigate (sweeps / geometry)
- nppiFilter/sweep/32f_C1R element 0 (32f 2D filter value).
- nppiFilter/geom even kernel sizes (kevw/kevh), negative kernel dims and
  negative dst step.
- nppiFilterRowBorder/sweep 8u C4R and AC4R element 0.
- nppiSumWindow row/column sweep values (window definition / clipping).
