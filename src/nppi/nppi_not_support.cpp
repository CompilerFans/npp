#include "npp.h"
#include <cstdio>
#include <cuda_runtime.h>

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
