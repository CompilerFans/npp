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

## Addendum (2026-11-01, after the executor landed)

The generic executors (`nppi_warp_perspective_executor.{h,cu}` and
`nppi_rotate_executor.{h,cu}`) plus `nppi_warp_sampling.cuh` implement the
following model, re-validated against the recorded goldens with a standalone
mcMPP-side build.  It refines the notes above in four places.

### Write region, single-image forward vs batch

* `nppiWarpPerspective`/`Quad` (single image): the row gate above does **not**
  exist.  Per destination row the kernel computes `v_row = v(srcROI.x, y)` and
  the x span `[ceil(Fx(srcROI.x, v_row)), floor(Fx(srcROI.x+w-1, v_row))]`, then
  applies the per-pixel gate.  Evidence: `fwd/8u_P3R/persp_subroi_cubic` writes
  dst row 3 although `v_row = 2.23 < srcROI.y = 3`.
* `nppiWarpPerspectiveBatch` keeps the extra row gate (`v_row < srcROI.y ||
  v_row > srcROI.y + srcROI.h` drops the row).  Evidence: batch `8u_C1R/n3`
  image 2 (kPersp) must skip dst row 1 (`v_row = -0.08`).
* Backward remains per-pixel only (measured before, unchanged).

### Interpolation

* NN: `floor(u + 0.5)`, no clamping surprises.
* LINEAR: plain bilinear on taps `floor(u)/floor(v)`.
* CUBIC: four-point **cubic Lagrange** (not Keys/cubic convolution) on taps
  `floor(u)-1 .. floor(u)+2`.  With `d = u - floor(u)` the 1-D weights are
  `L(-1) = -d(d-1)(d-2)/6`, `L(0) = (d+1)(d-1)(d-2)/2`,
  `L(+1) = -(d+1)d(d-2)/2`, `L(+2) = (d+1)d(d-1)/6`; the sweep in
  `fwd/32f_P3R/cubw_x*` fits these polynomials to 5e-7.
* Interpolation taps clamp to the **image** bounds (`0 .. oSize-1`), not to the
  ROI (`fwd/32f_P3R/persp_subroi_cubic` is exact only with image clamping).
* Integer destinations round with `floor(v + 0.5)` and saturate; Npp16f rounds
  through `__float2half`.

### Quad

* The destination write region is the image of the **source quadrilateral**
  (half-plane test of the inverse-mapped `(u, v)`), intersected with
  `den > 0`, the image bounds and the destination ROI.
* Pixels whose centre lands exactly on a projected edge/vertex are decided by
  the last bit of NVIDIA's coefficient arithmetic: with integer quads two
  `kQuadPersp*` vertex pixels are skipped while the `kQuadSame*` vertex pixels
  are written.  The probe quads are therefore shifted by `(0.31, 0.17)` (both
  quads together, so every mapping is unchanged); with that offset a float32
  half-plane test matches all recorded quad goldens exactly.

### Rotate (`nppiRotate`)

* dst -> src mapping is evaluated in **float32**:
  `fx = ca*(x-nShiftX) - sa*(y-nShiftY)`, `fy = sa*(x-nShiftX) + ca*(y-nShiftY)`
  with `ca/sa = (float)cos/sin(angle*pi/180)`.  The `angle30` row-11 pixels that
  land on exactly 1.5 only reproduce in single precision.
* A destination pixel is written iff `srcROI.x - 0.5 <= fx <= srcROI.x + w - 1`
  (same in y); outside pixels stay **untouched** (the rotate kernel does not
  fill), AC4 keeps its alpha channel, and the three interpolation modes behave
  exactly like the perspective ones.
* Contract: empty source size or empty destination ROI is a no-op
  (`NPP_NO_ERROR`), `srcROI.x/y < 0` is `NPP_SIZE_ERROR`, an empty/negative
  source ROI is `NPP_WRONG_INTERSECTION_QUAD_WARNING`, and a step that is not a
  multiple of the largest power of two dividing `channels*sizeof(T)` is
  `NPP_NOT_EVEN_STEP_ERROR` (the 32f AC4 probes use 392-byte steps).

### Contract codes (measured, per family)

| probe | forward/back planar | quad | batch | rotate |
|---|---|---|---|---|
| null pointer | `NPP_NULL_POINTER_ERROR` | `NULL_POINTER` (after the quads) | `NULL_POINTER` for a null list | `NULL_POINTER` |
| src step 0 | `NPP_NO_ERROR` (not validated) | `NPP_NO_ERROR` | n/a (in the list) | `NPP_STEP_ERROR` |
| dst step 0 | `NPP_STEP_ERROR` | `NPP_STEP_ERROR` | n/a | `NPP_STEP_ERROR` |
| empty src size | `NPP_SIZE_ERROR` | `NPP_SIZE_ERROR` | `NPP_RECTANGLE_ERROR` | `NPP_NO_ERROR` |
| empty src ROI | `NPP_RECTANGLE_ERROR` | `NPP_WRONG_INTERSECTION_QUAD_WARNING` | `NPP_RECTANGLE_ERROR` | `NPP_WRONG_INTERSECTION_QUAD_WARNING` |
| empty dst ROI | `NPP_NO_ERROR` | `NPP_WRONG_INTERSECTION_QUAD_WARNING` | `NPP_RECTANGLE_ERROR` | `NPP_NO_ERROR` |
| negative ROI origin | `NPP_SIZE_ERROR` | not validated | `NPP_SIZE_ERROR` | `NPP_SIZE_ERROR` |
| ROI outside the image | `NPP_NO_ERROR` | `NPP_NO_ERROR` | `NPP_NO_ERROR` | `NPP_NO_ERROR` |
| singular coefficients | fwd `NPP_COEFFICIENT_ERROR`, back `NPP_NO_ERROR` | `NPP_QUADRANGLE_ERROR` (degenerate quads, checked first) | not checked | n/a |
| bad interpolation | `NPP_INTERPOLATION_ERROR` | `NPP_INTERPOLATION_ERROR` | `NPP_INTERPOLATION_ERROR` | `NPP_INTERPOLATION_ERROR` |
| other | – | – | `nBatchSize < 2` -> `NPP_BAD_ARGUMENT_ERROR`; `BatchInit` with 0 -> `NPP_CUDA_KERNEL_EXECUTION_ERROR` | – |

### Implementation pitfalls worth remembering

* Kernel parameters declared as arrays (`const T *pSrc[3]`, `float quad[8]`)
  decay to pointers: the device then dereferences a **host** pointer.  The
  plane pointers and the quad corners must travel inside small structs
  (`PlaneSet<T,P>`, `QuadF`) so the launch copies them by value.
  `DeviceMemory`'s uninitialised `ptr_` in the test harness caused the same
  class of sticky "illegal memory access" and is fixed as well.
