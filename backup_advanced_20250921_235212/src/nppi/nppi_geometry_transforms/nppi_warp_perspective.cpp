#include "npp.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// Implementation file

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiWarpPerspective_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpPerspective_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                               Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                               int eInterpolation, NppStreamContext nppStreamCtx);
}

// Implementation file
static inline NppStatus validateWarpPerspectiveInputs(const void *pSrc, NppiSize oSrcSize, int nSrcStep,
                                                      NppiRect oSrcROI, void *pDst, int nDstStep, NppiRect oDstROI,
                                                      const double aCoeffs[3][3], int eInterpolation) {
  // Check pointers
  if (!pSrc || !pDst || !aCoeffs) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Check step sizes
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // Check source size
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check ROI sizes
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0 || oDstROI.width <= 0 || oDstROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  // Check ROI bounds
  if (oSrcROI.x < 0 || oSrcROI.y < 0 || oSrcROI.x + oSrcROI.width > oSrcSize.width ||
      oSrcROI.y + oSrcROI.height > oSrcSize.height) {
    return NPP_RECTANGLE_ERROR;
  }

  // Check interpolation mode
  if (eInterpolation != NPPI_INTER_NN && eInterpolation != NPPI_INTER_LINEAR && eInterpolation != NPPI_INTER_CUBIC) {
    return NPP_INTERPOLATION_ERROR;
  }

  // Check perspective matrix - the bottom row should not make w=0
  // We check if the matrix would cause division by zero for any destination pixel
  // This is a simplified check - a more thorough check would validate for all pixels
  if (fabs(aCoeffs[2][2]) < 1e-10) {
    return NPP_COEFFICIENT_ERROR;
  }

  return NPP_SUCCESS;
}

// Implementation file
NppStatus nppiWarpPerspective_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// Implementation file
NppStatus nppiWarpPerspective_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// Implementation file
NppStatus nppiWarpPerspective_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpPerspectiveInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                                   eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpPerspective_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpPerspective_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[3][3],
                                      int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpPerspective_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}