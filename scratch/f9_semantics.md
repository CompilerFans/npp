# f9 measured semantics (RTX 5090, NVIDIA NPP 13.4, sm_120)

Probes: scratch/probe_f9.cu, probe_f9b.cu, probe_f9c.cu, probe_f9d.cu
Outputs: scratch/remote_out/*.out (all pulled and read)
Baseline for ALL 11 f9 families. Semantics below were measured, not taken from docs.

## 1. Validation order / status codes (filter kernels)

Standard families (Box, BoxBorder, Max, Min, MaxBorder, MinBorder, Median, MedianBorder):
- null pSrc/pDst -> NPP_NULL_POINTER_ERROR (see f9b_valid.out for exact precedence)
- negative ROI dims -> NPP_SIZE_ERROR; zero-dim ROI -> see per-family notes below
- step: unsigned-too-small -> NPP_STEP_ERROR first, then alignment check
  dstep % (es * (ch==4 ? 4 : 1)) != 0 -> NPP_NOT_EVEN_STEP_ERROR  (standard families)
- negative mask dims -> NPP_MASK_SIZE_ERROR
- anchor outside mask -> NPP_ANCHOR_ERROR

Median (plain): NO dstep alignment check (only too-small).
Advanced: NO too-small check (except W==0 && H>0 -> always STEP); alignment is
  dstep % min(es,4) (i.e. mod element size but capped at 4 bytes).

## 2. Border types

- NppiBorderType values: UNDEFINED/NONE=0, CONSTANT=1, REPLICATE=2, WRAP=3, MIRROR=4.
- BoxBorder additionally ACCEPTS value 5 with OK (measured); others reject it
  (NPP_BORDER_REPLICATION_ERROR? -> see f9b_mirror.out for exact code).
- MaxBorder/MinBorder: REPLICATE accepted; CONST/WRAP acceptance NOT measured yet
  (S2-S6 outputs lost) -> re-probe needed (f9e).
- MedianBorder: REPLICATE + MIRROR accepted only.
- Border GetBufferSize accepts border types 0/1/3/7 with OK (7 also OK!).

## 3. AC4 alpha semantics (measured, box/max/min/median, plain and border, 8u and 16s)

- C4 variants filter the alpha channel like any other channel.
- AC4 variants leave dst alpha UNTOUCHED (dst alpha keeps its pre-call value,
  probe init 171 = 0xAB).

## 4. Border/offset model (confirmed for standard border families)

Image of oSrcSize with origin at pSrc - oSrcOffset (i.e. image pixel (i,j) lives at
pSrc[(j-offy)*sw + (i-offx)]). dst(x,y) filters the window around image pixel
(x+offx, y+offy) with mask oMaskSize and anchor oAnchor; reads outside the image
follow eBorderType (replicate/mirror/wrap measured; constant unmeasured for these
families).

## 5. Median / MedianBorder GetBufferSize (host function, pBuffer size in bytes)

value = 49152 * esize * maskW * maskH   when the "buffered" kernel path is selected,
      = 0                               otherwise.
Independent of ROI size. esize = element size (1/2/2/4 for 8u/16u/16s/32f).

Validation (measured):
- negative ROI -> NPP_SIZE_ERROR, no write to pBufferSize
- negative mask dim -> NPP_SIZE_ERROR (NOT mask error)
- zero-dim ROI -> NPP_OK, buffer NOT written (plain); Border zero-dim ROI unmeasured
- zero-dim mask -> NPP_OK; plain leaves buffer untouched, Border writes 0
- null pBufferSize -> SEGFAULT (do not test)

Exact dispatch (buffered vs not), 8u C1 (EXACT on all 1600 grid cells + specials):
- plain Median 8u C1:      buffered iff (w%2==1 && h%2==1) ? (w*h > 256) : ((w+1)*h > 190)
- MedianBorder 8u C1 (REPLICATE): buffered iff w*h > 192
Remaining 22 configs (dtype x ch x plain/border): need f9e grid probe.

## 6. nppiFilterBoxBorderAdvanced (13 _Ctx functions) — QUIRKY, decode incomplete

Signature (measured usage):
  nppiFilterBoxBorderAdvanced_<t>_CnR_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset,
      pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, NPP_BORDER_REPLICATE, pBuffer, ctx)
GetBufferSize: nppiFilterBoxBorderAdvancedGetDeviceBufferSize(oROI, nChannels, &sz)
  -> 204 bytes for 7x7 C1, 264 for 8x8 C1 (function exists; _64 variant for 64f).
  Kernel does not write past bsz (canary intact).

Measured behavior:
- Output = (sum of exactly 9 taps)/9 for a 3x3 mask (denominator = mw*mh), linear
  in source values, deterministic, independent of pBuffer prefill.
- Interior cells read a pure 3x3 window, but at a SHIFTED position relative to the
  standard border model (see rmap analysis: R1 off=(2,3) sw=sh=16 ROI 7x7 mask 3x3
  a=(1,1): interior window cols [x+offx+W-ax, +mw-1] x rows [y+offy-ay-1, +mh-1]).
- Edge cells: taps form 3 groups of 3 consecutive FLAT indices; group start offsets
  step by +W (ROI width) or +sw (source stride) or +0 depending on cell; negative
  flat indices wrap mod sw; positions >= sw replicate-clamp to the edge column.
- I4 (64x64 img, ROI 48, off (8,8), imp (31,31)): clean standard 3x3 landing at
  dst(22..24,22..24). I5 (96x96, ROI 64, off (8,8), imp (39,39)): lands ONLY column
  dst x=32, rows 30..32. I6 (mask 5x5, imp (39,39)=250): 2-col x 5-row block
  (cols 32-33, rows 29-33), tap weight 10 = 250/25 each.
-=> The tap POSITION formula is an NVIDIA implementation quirk (pointer arithmetic
  aliasing ROI width vs source stride). NOT yet decoded. Needs probe f9e with
  1x1 / 3x1 / 1x3 masks, sw==W cases, other geometries, other border types, 16u/32f,
  C3/C4 before implementing.

## 7. Open items (to measure in probe f9e)

- Median/MedianBorder GetBufferSize '#'/'.' grids for remaining 22 configs +
  value vs border type for Border variant.
- Median filter semantics: even-count convention (lower/average), buffered vs
  non-buffered path equality, AC4 handling (already known: alpha untouched).
- MaxBorder/MinBorder CONST/WRAP acceptance (lost S2-S6).
- null pSrc/pDst exact statuses for every family (partial data in f9b_valid.out).
- negative ROI for border families; mask > ROI for Max/Min; BoxBorder src-step
  validation; Advanced C4 alignment; constant border value for BoxBorder
  (aAval? no aCoeff — CONSTANT border uses 0? unmeasured).

## 8. Third continuation: direct executor-vs-NVIDIA verification (2026-11-01)

`scratch/verify_f9.cu` links `nppi_filter_window_executor.cu` next to NVIDIA
NPP on the 5090 (the kernels are plain CUDA) and compares every family:
51,622 cases, 0 failures after the fixes below.  `scratch/probe_f9f..k.cu`
are the targeted follow-ups.

Corrections / additions to the earlier notes:

- nppiFilterBox validation: a **positive** ROI accepts masks larger than the
  ROI (no mask>ROI check at all).  An empty ROI is OK only for the 8u_C1R and
  64f_C1R instantiations; every other dtype/layout returns MASK_SIZE.  Negative
  masks stay OUT_OFF_RANGE (SIZE when the ROI is empty).
- nppiFilterBoxBorder: empty ROI is OK only for 8u_C1R (MASK_SIZE otherwise).
  MIRROR (4) and the undocumented value 5 are only accepted for 3x3 and 5x5
  masks (other masks -> MASK_SIZE) and the 32f_AC4R layout rejects them with
  NOT_SUPPORTED_MODE.
- median GetBufferSize (plain): empty ROI **and** empty mask leave the output
  size untouched (they do not write 0); negative ROI/mask -> SIZE, no write.
  Border variant writes 0 first and then reports SIZE for negative inputs.
- median even-count convention: the small-mask unbuffered kernels differ by
  mask shape for signed 16s (some shapes truncate toward zero, some floor);
  the 16x16 buffered path matches trunc((a+b)/2).  Differential tests therefore
  use odd mask dimensions for value comparisons (even 2D masks are not a
  reproducible contract).
- destination steps used by the differential tests are padded to 16 bytes;
  NVIDIA's small-mask median kernels misbehave on narrower steps without
  reporting an error.
- GetBufferSize is ROI-independent (verified for ROI 7x7, 32x32 and
  ROI == mask); the plain dispatch table is transcribed for masks up to 40x40.

## 9. Deliberately not implemented

- `nppiFilterBoxBorderAdvanced` (13 Ctx functions): the 5090 output follows a
  pointer-aliasing quirk (ROI width vs source stride) that is not decoded; the
  family is left missing and excluded from the package wrappers.
