# Wave Size (32/64) Compatibility Audit

Date: 2026-09-10
Scope: full sweep of `src/` for warp-width coupling: `__shfl*`/`__ballot`/`__syncwarp`/`__activemask`, `warpSize`, `WARP_SIZE`, shared-memory reductions, fixed block-size assumptions.

## Conclusion

No runtime bug under the current CUDA configuration (warp = 32, all launches 256 threads).
The only warp-coupled file in the entire codebase is `src/nppi/nppi_statistics_functions/nppi_mean_stddev.cu`.
All other modules (histogram, transpose, npps_sum, npps_integral, filtering, morphology) use
`blockDim.x / 2` shared-memory tree reductions or atomics, which are warp-width agnostic.

The warp=64 branch exists as dead code (`#if 1` pins warp=32) and has never been validated on
warp=64 hardware. See finding F1.

## Coupling inventory (nppi_mean_stddev.cu only)

- 1 `warpReduceSum` helper (line 12): `__shfl_down_sync` loop over `WARP_SIZE/2`
- 1 `blockReduceSum` helper (line 21): lane/wid decomposition + shared staging
- 17 `__shared__ double shared[WARP_SIZE]` arrays in kernels
- 33 `blockReduceSum(...)` call sites
- 14 kernel launches, all hardcoded `blockSize = 256`

Other hits are false positives: `nppi_warp_perspective.cpp` (function naming),
`tools/check_coverage.py` (wrapper-name parsing).

## Findings

### F1. Dead warp=64 branch with a real compile warning (medium)

`nppi_mean_stddev.cu:5-12`:

```c
#if 1
#define WARP_SIZE 32
#define SHFL_MASK 0xFFFFFFFF
#else
#define WARP_SIZE 64
#define SHFL_MASK 0xFFFFFFFFFFFFFFFFULL
#endif
```

Compiling with the `#else` branch enabled produces:

```
warning #69-D: integer conversion resulted in truncation
  __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, offset)
```

Root cause: CUDA `__shfl_*_sync` takes an `unsigned` (32-bit) mask. The 64-bit mask constant is
unrepresentable, so the 64 branch has never been usable as written. It is a porting placeholder
(e.g. MUSA warp=64), not tested support.

Recommended: if no warp=64 port is planned soon, delete the dead branch. If kept, the mask for a
warp=64 platform must come from that platform's shuffle API, not a wider literal.

### F2. blockReduceSum is correct for both widths at block=256, but has a hidden precondition (low)

Verified by line-by-line reasoning for a 256-thread block:

- warp=32: 8 warps; `shared[0..7]` written; read guard `tid < blockDim.x / WARP_SIZE` (tid < 8)
  reads `shared[lane]` with lane = tid % 32 = tid. Correct.
- warp=64: 4 warps; `shared[0..3]` written; guard `tid < 4`; lanes 4..63 contribute 0. Correct.

Precondition: every launch using `blockReduceSum` must use a block size that is a multiple of
`WARP_SIZE`. Today all 14 launches hardcode 256 (a multiple of both 32 and 64), so the code is
safe. But if a future launch uses a non-multiple (e.g. 96), warp=64 would silently drop the tail
warps' partial sums (`blockDim.x / WARP_SIZE` floors) - a wrong-result, no-crash failure mode.

Recommended: document the multiple-of-WARP_SIZE requirement at `blockReduceSum`, or add a
debug-mode runtime check.

### F3. Sequential blockReduceSum reuse relies on the trailing barrier (low)

`nppiMean_CxR_kernel_impl` reduces up to 4 channel accumulators back-to-back on the same
`shared` array, and `finalReductionMasked_kernel` chains three reductions. Correctness between
consecutive calls depends on the `__syncthreads()` at the end of `blockReduceSum`, which prevents
the next call's `shared[wid] = val` write from racing the previous call's read. The barrier is
present; this is only a fragile invariant worth a comment at the helper.

## Non-coupled patterns audited (no action needed)

- `npps_sum.cu`, `npps_integral.cu`: `sdata[256]` + `blockDim.x / 2` tree reduction / scan
- `nppi_histogram.cu`: `extern __shared__` + atomics (96 sites)
- `nppi_transpose.cu`: fixed 32x32 tile with bank-conflict padding (tile size is unrelated to
  warp width)
- `nppi_distance_transform.cu`: 16x16 blocks, no shared reductions
- `nppi_compressed_markerLabels.cu`, `nppi_watershed.cu`: atomics only
