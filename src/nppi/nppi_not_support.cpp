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













/** @} */

/** @} image_histogrameven */

/** @defgroup image_histogramrange HistogramRange
 * Primitives for computing the histogram of an image within specified ranges.
 * @{
 *
 */


















