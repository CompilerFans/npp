#include "npp.h"
#include <cstdio>
#include <cuda_runtime.h>

NppStatus nppiSumWindowRow_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiSumWindowRow_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

// NppStatus
// nppiAndC_32s_C3R(const Npp32s * pSrc1, int nSrc1Step, const Npp32s  aConstants[3],
//		 Npp32s * pDst,  int nDstStep,  NppiSize oSizeROI){
//    return NPP_NOT_IMPLEMENTED_ERROR;
// }
// NppStatus
// nppiAndC_16u_C3R(const Npp16u * pSrc1, int nSrc1Step, const Npp16u  aConstants[3],
//		Npp16u * pDst,  int nDstStep,  NppiSize oSizeROI){
//   return NPP_NOT_IMPLEMENTED_ERROR;
// }

NppStatus nppiRShiftC_8s_C1R(const Npp8s *pSrc1, int nSrc1Step, const Npp32u nConstant, Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiRShiftC_8s_C1R_Ctx(const Npp8s *pSrc1, int nSrc1Step, const Npp32u nConstant, Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRShiftC_8s_C3R(const Npp8s *pSrc1, int nSrc1Step, const Npp32u aConstants[3], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiRShiftC_8s_C3R_Ctx(const Npp8s *pSrc1, int nSrc1Step, const Npp32u aConstants[3], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRShiftC_8s_C4R(const Npp8s *pSrc1, int nSrc1Step, const Npp32u aConstants[4], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiRShiftC_8s_C4R_Ctx(const Npp8s *pSrc1, int nSrc1Step, const Npp32u aConstants[4], Npp8s *pDst, int nDstStep,
                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiSumWindowColumn_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiSumWindowColumn_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
// NppStatus nppiCopy_32f_P3C3R(const Npp32f * const aSrc[3], int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize
// oSizeROI){
//
//	return NPP_NOT_IMPLEMENTED_ERROR;
// }

/**
 * Four-channel 8-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation,
                                     NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 8-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[3],
                                int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 8-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_8u_P4R_Ctx(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst[4],
                                int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Single-channel 16-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                  int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 16-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_P3R_Ctx(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 16-bit unsigned affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16u_P4R_Ctx(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Single-channel 32-bit signed affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_C1R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 32-bit signed affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_C3R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit signed affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_C4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit signed affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_AC4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32s *pDst,
                                  int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit signed affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_P3R_Ctx(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit signed affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32s_P4R_Ctx(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Single-channel 16-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16f_C1R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16f_C3R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_16f_C4R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_16f_C4R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 32-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating-point affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                  int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32f_P3R_Ctx(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_32f_P4R_Ctx(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Single-channel 64-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_C1R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_C1R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 64-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_C3R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_C3R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 64-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_C4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_C4R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 64-bit floating-point affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePackedPixelParameters">Common parameters for
 * nppiWarpAffine packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_AC4R_Ctx(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp64f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_AC4R(const Npp64f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp64f *pDst,
                                  int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 64-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_P3R_Ctx(const Npp64f *aSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *aDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_P3R(const Npp64f *aSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp64f *aDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 64-bit floating-point affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffinePlanarPixelParameters">Common parameters for
 * nppiWarpAffine planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffine_64f_P4R_Ctx(const Npp64f *aSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp64f *aDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffine_64f_P4R(const Npp64f *aSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                 Npp64f *aDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                 int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 8-bit unsigned image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate
 * functions</a>.
 *
 */
NppStatus nppiRotate_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRotate_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                             int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit unsigned image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate
 * functions</a>.
 *
 */
NppStatus nppiRotate_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                                  int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                  int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRotate_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp16u *pDst,
                              int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                              int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float image rotate ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonRotateParameters">Common parameters for nppiRotate
 * functions</a>.
 *
 */
NppStatus nppiRotate_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                  int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                  int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRotate_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                              int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                              int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 8-bit unsigned image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 8-bit unsigned in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit unsigned image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16u_AC4R(const Npp16u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oROI,
                              NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit unsigned in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 1 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C1R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 1 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C3R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit signed image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                                 NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                             NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit signed in place image mirror.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_C4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                  NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_C4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit signed image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_AC4R(const Npp16s *pSrc, int nSrcStep, Npp16s *pDst, int nDstStep, NppiSize oROI,
                              NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 16-bit signed in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_32s_AC4R_Ctx(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_32s_AC4R(const Npp32s *pSrc, int nSrcStep, Npp32s *pDst, int nDstStep, NppiSize oROI,
                              NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit signed in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_32s_AC4IR_Ctx(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_32s_AC4IR(Npp32s *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                                  NppiAxis flip, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_32f_AC4R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oROI,
                              NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float in place image mirror not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorParameters">Common parameters for nppiMirror
 * functions</a>.
 *
 */
NppStatus nppiMirror_32f_AC4IR_Ctx(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirror_32f_AC4IR(Npp32f *pSrcDst, int nSrcDstStep, NppiSize oROI, NppiAxis flip) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/** @} image_mirror */

/** @defgroup mirror_batch MirrorBatch
 *  Mirrors batches of images horizontally, vertically or diagonally.
 *
 * MirrorBatch generally takes the same parameter list as Mirror except that there is a list of N instances of those
 * parameters (N > 1) and that list is passed in device memory.  A convenient data structure is provided that allows for
 * easy initialization of the parameter lists.  The only restriction on these functions is that there is one single ROI
 * and a single mirror flag which are applied respectively to each image in the batch.  The primary purpose of this
 * function is to provide improved performance for batches of smaller images as long as GPU resources are available.
 * Therefore it is recommended that the function not be used for very large images as there may not be resources
 * available for processing several large images simultaneously.
 *
 * <h3><a name="CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch non-inplace and inplace functions
 * include:</a></h3>
 *
 * \param oSizeROI \ref roi_specification.
 * \param flip Specifies the axis about which the images are to be mirrored.
 * \param pBatchList Device memory pointer to nBatchSize list of NppiMirrorBatchCXR structures.
 * \param nBatchSize Number of NppiMirrorBatchCXR structures in this call (must be > 1).
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes, \ref mirror_error_codes
 *
 * @{
 *
 */

/**
 * 1 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C1R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                      NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C1R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 1 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C1IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C1IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C3R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                      NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C3R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C3IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C3IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                      NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float in place image mirror batch.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_C4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_C4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float image mirror batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_AC4R_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_AC4R(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 4 channel 32-bit float in place image mirror batch not affecting alpha.
 *
 * For common parameter descriptions, see <a href="#CommonMirrorBatchParameters">Common parameters for nppiMirrorBatch
 * functions</a>.
 *
 */
NppStatus nppiMirrorBatch_32f_AC4IR_Ctx(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList,
                                        int nBatchSize, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiMirrorBatch_32f_AC4IR(NppiSize oSizeROI, NppiAxis flip, NppiMirrorBatchCXR *pBatchList, int nBatchSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * One-channel 32-bit signed image RectStdDev.
 *
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param pSqr \ref destination_image_pointer.
 * \param nSqrStep \ref destination_image_line_step.
 * \param pDst \ref destination_image_pointer.
 * \param nDstStep \ref destination_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param oRect rectangular window
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiRectStdDev_32s32f_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep,
                                        Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiRectStdDev_32s32f_C1R(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep, Npp32f *pDst,
                                    int nDstStep, NppiSize oSizeROI, NppiRect oRect) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned integer backwards affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for
 * nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 8-bit unsigned integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                        int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 8-bit unsigned integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_8u_P4R_Ctx(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                        int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned integer backwards affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for
 * nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 16-bit unsigned integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_16u_P3R_Ctx(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 16-bit unsigned integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_16u_P4R_Ctx(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit signed integer backwards affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for
 * nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32s_AC4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit signed integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32s_P3R_Ctx(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit signed integer backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32s_P4R_Ctx(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating-point backwards affine warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPackedPixelParameters">Common parameters for
 * nppiWarpAffineBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit floating-point backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32f_P3R_Ctx(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit floating-point backwards affine warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpAffineBackPlanarPixelParameters">Common parameters for
 * nppiWarpAffineBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpAffineBack_32f_P4R_Ctx(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpAffineBack_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned integer backwards perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 8-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_8u_P4R_Ctx(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned integer backwards perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_16u_P3R_Ctx(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst[3], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 16-bit unsigned integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_16u_P4R_Ctx(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst[4], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit signed integer backwards perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32s_AC4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32s_P3R_Ctx(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst[3], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit signed integer backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32s_P4R_Ctx(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst[4], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating-point backwards perspective warp, ignorning alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPackedPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32f_P3R_Ctx(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst[3], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit floating-point backwards perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectiveBackPlanarPixelParameters">Common parameters
 * for nppiWarpPerspectiveBack planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspectiveBack_32f_P4R_Ctx(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst[4], int nDstStep, NppiRect oDstROI,
                                              const double aCoeffs[3][3], int eInterpolation,
                                              NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspectiveBack_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned integer perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_8u_AC4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_8u_AC4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                      int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_8u_P3R_Ctx(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_8u_P3R(const Npp8u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 8-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_8u_P4R_Ctx(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_8u_P4R(const Npp8u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp8u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                     int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned integer perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16u_AC4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16u_AC4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                       Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                       int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16u_P3R_Ctx(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16u_P3R(const Npp16u *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 16-bit unsigned integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16u_P4R_Ctx(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16u_P4R(const Npp16u *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16u *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit signed integer perspective warp, igoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32s_AC4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32s_AC4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                       Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                       int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32s_P3R_Ctx(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32s_P3R(const Npp32s *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit signed integer perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32s_P4R_Ctx(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32s_P4R(const Npp32s *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32s *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Single-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16f_C1R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16f_C1R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16f_C3R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16f_C3R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_16f_C4R_Ctx(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_16f_C4R(const Npp16f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp16f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating-point perspective warp, ignoring alpha channel.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePackedPixelParameters">Common parameters for
 * nppiWarpPerspective packed pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32f_AC4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                           Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                           int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32f_AC4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                       Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                       int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel planar 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32f_P3R_Ctx(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32f_P3R(const Npp32f *pSrc[3], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst[3], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel planar 32-bit floating-point perspective warp.
 *
 * For common parameter descriptions, see <a href="#CommonWarpPerspectivePlanarPixelParameters">Common parameters for
 * nppiWarpPerspective planar pixel functions</a>.
 *
 */
NppStatus nppiWarpPerspective_32f_P4R_Ctx(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiWarpPerspective_32f_P4R(const Npp32f *pSrc[4], NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst[4], int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 8-bit unsigned planar color not in place forward gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGammaFwd_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiGammaFwd_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 8-bit unsigned planar color in place forward gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGammaFwd_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiGammaFwd_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 8-bit unsigned planar color not in place inverse gamma correction.
 *
 * \param pSrc source planar pixel format image pointer array.
 * \param nSrcStep source planar pixel format image line step.
 * \param pDst destination planar pixel format image pointer array.
 * \param nDstStep destination planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGammaInv_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiGammaInv_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * 3 channel 8-bit unsigned planar color in place inverse gamma correction.
 *
 * \param pSrcDst in place planar pixel format image pointer array.
 * \param nSrcDstStep in place planar pixel format image line step.
 * \param oSizeROI \ref roi_specification.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 */
NppStatus nppiGammaInv_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiGammaInv_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Three-channel 8-bit unsigned HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                       int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                       NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3], int nLevels[3],
                                   Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
NppStatus nppiHistogramEven_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}


/**
 * Three-channel 16-bit unsigned HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16u_C3R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                        int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16u_C4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                    int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16u_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit signed HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16s_C3R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramEven.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                        int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16s_C4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                    int nLevels[4], Npp32s nLowerLevel[4], Npp32s nUpperLevel[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramEven ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramEven_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramEven_16s_AC4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     int nLevels[3], Npp32s nLowerLevel[3], Npp32s nUpperLevel[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/** @} */

/** @} image_histogrameven */

/** @defgroup image_histogramrange HistogramRange
 * Primitives for computing the histogram of an image within specified ranges.
 * @{
 *
 */

/** @name HistogramEven
 * The histogram is computed according to the ranges provided in \f$pLevels\f$.
 * The histogram \f$pHist[k]\f$ is defined as the total number of pixels that fall into the range:
 * \f$pLevels[k] <= pSrc(j, i) < pLevels[k+1]\f$. The number of the histogram bins is \f$nLevel - 1\f$.
 * The functions require additional scratch buffer for computations.
 * <h3><a name="CommonHistogramEvenParameters">Common parameters for nppiHistogramEven functions include:</a></h3>
 * \param pSrc \ref source_image_pointer.
 * \param nSrcStep \ref source_image_line_step.
 * \param oSizeROI \ref roi_specification.
 * \param pHist Pointer to array that receives the computed histogram.
 *      The array must be of size nLevels-1.
 * \param pLevels Pointer to array containing the level sizes of the bins.
        The array must be of size nLevels.
 * \param nLevels Number of levels in histogram.
 * \param pBuffer Pointer to appropriately sized (nppiHistogramRangeGetBufferSize_XX_XXX)
 *      scratch buffer.
 * \param nppStreamCtx \ref application_managed_stream_context.
 * \return \ref image_data_error_codes, \ref roi_error_codes
 * @{
 */

/**
 * Three-channel 8-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                        const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                    const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                        const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer,
                                        NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_8u_C4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                    const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 8-bit unsigned HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_8u_AC4R(const Npp8u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * One-channel 16-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                         const Npp32s *pLevels, int nLevels, Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_C1R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                     const Npp32s *pLevels, int nLevels, Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_C3R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_C4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                         const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_C4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                     const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit unsigned HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16u_AC4R(const Npp16u *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * One-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                         const Npp32s *pLevels, int nLevels, Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_C1R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist,
                                     const Npp32s *pLevels, int nLevels, Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_C3R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_C4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                         const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_C4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                     const Npp32s *pLevels[4], int nLevels[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 16-bit signed HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_16s_AC4R(const Npp16s *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32s *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Three-channel 32-bit floating point HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                         const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                     const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating point HistogramRange.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_32f_C4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                         const Npp32f *pLevels[4], int nLevels[4], Npp8u *pBuffer,
                                         NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_32f_C4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[4],
                                     const Npp32f *pLevels[4], int nLevels[4], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Four-channel 32-bit floating point HistogramRange ignoring alpha channel.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenParameters">Common parameters for
 * nppiHistogramEven functions</a>.
 */
NppStatus nppiHistogramRange_32f_AC4R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                          const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer,
                                          NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRange_32f_AC4R(const Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI, Npp32s *pHist[3],
                                      const Npp32f *pLevels[3], int nLevels[3], Npp8u *pBuffer) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Scratch-buffer size for nppiHistogramRange_16u_C1R.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenGetBufferSizeParameters">Common parameters for
 * nppiHistogramEvenGetBufferSize functions</a>.
 */
NppStatus nppiHistogramRangeGetBufferSize_16u_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRangeGetBufferSize_16u_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

/**
 * Scratch-buffer size for nppiHistogramRange_16s_C1R.
 * For common parameter descriptions, see <a href="#CommonHistogramEvenGetBufferSizeParameters">Common parameters for
 * nppiHistogramEvenGetBufferSize functions</a>.
 */
NppStatus nppiHistogramRangeGetBufferSize_16s_C1R_Ctx(NppiSize oSizeROI, int nLevels, int *hpBufferSize,
                                                      NppStreamContext nppStreamCtx) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}

NppStatus nppiHistogramRangeGetBufferSize_16s_C1R(NppiSize oSizeROI, int nLevels, int *hpBufferSize) {
  return NPP_NOT_IMPLEMENTED_ERROR;
}
