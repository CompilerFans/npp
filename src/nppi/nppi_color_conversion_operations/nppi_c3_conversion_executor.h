// c3 executor: NPPI YUV/YCbCr conversion families
//
// Host-visible surface (plain.cpp translation units): entry points are
// declared here and defined in nppi_c3_conversion_executor.cu.  The public
//
// Semantics baseline: NVIDIA NPP 13.4 (reference), captured by reference comparisons:
//   * All 8u color conversions are computed in single precision with C-style
//     truncation toward zero and saturation to [0, 255] (validated bit-exact
//     against NVIDIA reference outputs; fixed-point 16-bit emulation does NOT match the
//     inverse JPEG matrix, so float is the contract).
//   * ROI writes only inside oSizeROI; row tails keep their prior content.
//   * Parameter validation order: NULL pointer -> step <= 0 -> negative ROI,
//     mirroring the status codes recorded from the reference library.
#ifndef NPPI_C3_CONVERSION_EXECUTOR_H
#define NPPI_C3_CONVERSION_EXECUTOR_H

#include "npp.h"
#include <cuda_runtime.h>

namespace nppi {
namespace c3conv {

// --- JPEG full-swing YCbCr 4:4:4 (BT.601 full range) ----------------------
// RGB or BGR source (packed C3 or planar P3) -> planar YCbCr.
NppStatus rgbToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream);
NppStatus rgbToYCbCr444JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                cudaStream_t stream);
NppStatus bgrToYCbCr444JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream);
NppStatus bgrToYCbCr444JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                cudaStream_t stream);
// Planar YCbCr -> RGB or BGR (packed C3 or planar P3).
NppStatus yCbCr444ToRgbJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream);
NppStatus yCbCr444ToRgbJpegP3C3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream);
NppStatus yCbCr444ToBgrJpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream);
NppStatus yCbCr444ToBgrJpegP3C3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream);

// === JPEG 4:2:2 / 4:2:0 subsampled conversions ==============================
// bgr != 0 swaps the R/B interpretation of the packed/planar RGB source (or
// the R/B write order of the RGB destination).
NppStatus rgbToYCbCr420JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                              NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus rgbToYCbCr420JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                                NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus rgbToYCbCr422JpegP3(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                              NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus rgbToYCbCr422JpegC3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                                NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus yCbCr420ToRgbJpegP3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus yCbCr420ToRgbJpegP3C3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus yCbCr422ToRgbJpegP3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus yCbCr422ToRgbJpegP3C3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI, cudaStream_t stream, int bgr);

// === video-range (studio-swing) 4:2:2 conversions ===========================
// C2 4:2:2 layouts: YCbCr422 [Y0 Cb Y1 Cr], CbYCr422 [Cb Y0 Cr Y1].
// srcPp selects the packed source pixel pitch (3 = C3, 4 = AC4); bgr swaps
// the R/B channel interpretation; cbycr selects the C2 layout; dstPp selects
// the packed destination pitch (3 = C3, 4 = C4/BGRA with aAval in slot 3).
NppStatus rgbToYCbCr422C3C2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            cudaStream_t stream, int srcPp, int bgr, int cbycr);
NppStatus rgbToYCbCr422C3P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], const int aDstStep[3],
                            NppiSize oSizeROI, cudaStream_t stream, int srcPp, int bgr);
NppStatus rgbToYCbCr422P3C2(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                            cudaStream_t stream);
NppStatus yCbCr422ToRgbC2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                          cudaStream_t stream, int dstPp, int bgr, int cbycr, Npp8u aval);
NppStatus yCbCr422ToRgbC2P3(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                            cudaStream_t stream, int cbycr);
NppStatus yCbCr422ToRgbP3C3(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst, int nDstStep,
                            NppiSize oSizeROI, cudaStream_t stream, int bgr);
NppStatus yCbCr422ToCbYCr422(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             cudaStream_t stream);
// NVIDIA 13.4 quirk: this call always returns NPP_STEP_ERROR after the ROI
// checks and never writes the destination (see executor for details).
NppStatus rgbToCbYCr422GammaC3C2(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI);

} // namespace c3conv
} // namespace nppi

#endif // NPPI_C3_CONVERSION_EXECUTOR_H
