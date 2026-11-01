// c3 package executor: NPPI YUV/YCbCr/YCC/CMYK color-space conversions
// (nppi_color_conversion.h, docs/agent_packages/c3.md).
//
// Semantics baseline: NVIDIA NPP 13.4 (RTX 5090).  Coefficients were fitted
// from dense 8-bit channel ramps (test/unit/diff/test_nppi_c3_color_generated.cpp
// records the goldens) and are applied in single precision with C-style
// truncation towards zero and saturation to [0,255].
//
// Matrix ids:
//   0 BT.601 video range  (Y +16, chroma +128)
//   1 BT.709 video range  (Y +16, chroma +128)
//   2 BT.709 full range   (Y no offset, chroma +128, chroma matrix scaled)
#ifndef NPPI_C3_COLOR_EXECUTOR_H
#define NPPI_C3_COLOR_EXECUTOR_H

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace c3color {

// --- packed RGB/BGR (C3 or AC4) -> planar YUV 4:2:0 ----------------------
// bgr != 0 reads the source as B,G,R.  swapPlanes writes (Cr, Cb) instead of
// (Cb, Cr) for the YCrCb family.  Odd width or height reports
// NPP_DOUBLE_SIZE_WARNING (the odd edge is not converted).
NppStatus fwd420(const Npp8u *pSrc, int nSrcStep, int srcPp, int bgr, Npp8u *pDst[3], const int aDstStep[3],
                 NppiSize oSizeROI, cudaStream_t stream, int matrix, int swapPlanes);

// --- packed RGB/BGR (C3 or AC4) -> packed 4:2:2 C2 -----------------------
// cbycr != 0 writes [Cb Y0 Cr Y1], otherwise [Y0 Cr Y1 Cb] (YCrCb422).
// Chroma is the pair average; a trailing odd pixel keeps the un-averaged
// chroma of its pair (no warning).
NppStatus fwdC2(const Npp8u *pSrc, int nSrcStep, int srcPp, int bgr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                cudaStream_t stream, int matrix, int cbycr);

// --- planar RGB (P3) -> packed 4:2:2 YCrCb422 ----------------------------
NppStatus fwdP3C2(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                  cudaStream_t stream);

// --- planar YUV 4:2:0 -> packed RGB/BGR (C3/C4) --------------------------
// srcSwap reads pSrc[1] as Cr and pSrc[2] as Cb.  bgr != 0 writes B,G,R.
// dstPp 3 or 4 (slot 3 filled with nAval).
NppStatus inv420(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep, int dstPp, int bgr,
                 Npp8u nAval, NppiSize oSizeROI, cudaStream_t stream, int matrix, int srcSwap);

// --- planar 4:2:2 YCbCr -> packed BGR ------------------------------------
// Single source step for all three planes (709CSC family).
NppStatus inv422Planar(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, Npp8u nAval,
                       NppiSize oSizeROI, cudaStream_t stream);

// --- packed CbYCr422 -> packed BGR (BT.709 full range) -------------------
NppStatus invC2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, Npp8u nAval, NppiSize oSizeROI,
                cudaStream_t stream);

// --- PhotoYCC -------------------------------------------------------------
// Packed RGB/AC4 -> packed YCC (C3/AC4), and the inverse.  The alpha slot is
// copied through.
NppStatus rgbToYcc(const Npp8u *pSrc, int nSrcStep, int pp, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream, int bgr);
NppStatus yccToRgb(const Npp8u *pSrc, int nSrcStep, int pp, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                   cudaStream_t stream);

// --- JPEG CMYK / YCCK -----------------------------------------------------
// 4 planar planes -> packed RGB/BGR or planar RGB.  K multiplies the
// inverted CMY channels (multiplied by 1/255 in single precision).
NppStatus cmykToRgb(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep, int dstPp, int bgr,
                    NppiSize oSizeROI, cudaStream_t stream);
NppStatus cmykToRgbPlanar(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                          NppiSize oSizeROI, cudaStream_t stream, int bgr);
// JPEG YCCK (BT.601) -> CMYK; the reference writes the inverted red channel
// into all four planes (Cb and K are ignored).
NppStatus ycckToCmyk(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI,
                     cudaStream_t stream);

} // namespace c3color
} // namespace nppi

#endif // NPPI_C3_COLOR_EXECUTOR_H
