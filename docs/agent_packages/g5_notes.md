# g5 reverse-engineering notes (measured on the RTX 5090, NPP 13.4)

## WarpPerspective / WarpPerspectiveBack / Batch: write-region model

Goldens under `artifacts/5090/npp_records/nppiWarpPerspective{,Batch}/` show that
NPP does *not* write every dst pixel whose inverse map lands in the source ROI.
The measured rule (validated exactly on 8u/16f/32f C1R, C3R, C4R, AC4R batch n3/n2
and the planar fwd P3R cases):

For a kernel with dst->src matrix `M` (fwd/batch: `M = true inverse of aCoeffs`,
back: `M = aCoeffs`) and forward map `F = M^-1`:

1. Row gate (per dst row `y` inside the dst ROI):
   `v_row = v(dstROI.x, y)`  (src v of the dst ROI's *left edge* pixel).
   The whole row is skipped when `v_row < srcROI.y` or `v_row > srcROI.y+srcROI.h-1`.
   * This kills the kPersp "row 1 strip": at dst y=1, v(0,1) = -0.0799 < 0.
2. Per-row x span (evaluated at the *row-gate* v, not the pixel's own v):
   `xl = Fx(srcROI.x, v_row)`, `xr = Fx(srcROI.x + srcROI.w - 1, v_row)`;
   pixels fill `ceil(xl) <= x <= floor(xr)` intersected with the dst ROI.
   * This is what skips dst(18,5) but keeps dst(18,6) for kPersp
     (u(18,5)=10.964 passes the pixel rule, but the row span ends at 17.99).
3. Per-pixel gate: `den > 0` and
   `srcROI.x <= u <= srcROI.x+srcROI.w-1` and `srcROI.y <= v <= srcROI.y+srcROI.h-1`
   (continuous, *inclusive* bounds - i.e. up to the last source pixel, not w).
   * scale2 writes x<=22,y<=16 (u<=11.0, v<=8.0); x=23/y=17 stay untouched.
   * kPersp row 8 stops at x=4 (v(4,8)=7.9975 <= 8 < 8.0611=v(5,8)).
   * shift_subroi proves the bounds are the *src ROI rect in absolute
     coordinates*, not the image size: region = x in [8,21], y in [2,11].

Interpolation mode does not change the region (NN/LINEAR/CUBIC identical).

## Sampling (NN)
sample = src[floor(v+0.5)*sStep/els + floor(u+0.5)] (round half up), i.e. u=4.5 -> 5.

## Batch semantics
* `nppiWarpPerspectiveBatchInit` writes `aTransformedCoeffs = true inverse of
  pCoeffs` (adjugate/det in f64, preserves -0.0) into each device entry.
* The batch kernel warps dst->src with `aTransformedCoeffs`, same "fwd" model.
* The post-call `aTransformedCoeffs` is unchanged by the batch warp call.
* 16f sources use `make_uniform<Npp16f>` = `to_npp16f(unit_float()*4-2)`
  (unit_float uses the top 24 mantissa bits, not unit_double).

Untouched dst keeps the 0xA5 prefill (16f: to_npp16f(0.645f)); note the 8u
*uniform* sources can collide with 0xA5, which punches holes in the "written"
mask (e.g. batch n2 kShift src[3*256+5]=165).
