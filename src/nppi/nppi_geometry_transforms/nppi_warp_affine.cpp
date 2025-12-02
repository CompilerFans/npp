#include "npp.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiWarpAffine_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffine_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffine_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                          Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                          int eInterpolation, NppStreamContext nppStreamCtx);
}

static inline NppStatus validateWarpAffineInputs(const void *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                                 void *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                                 int eInterpolation) {
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
  // // Check affine matrix determinant (for invertibility)
  // double det = aCoeffs[0][0] * aCoeffs[1][1] - aCoeffs[0][1] * aCoeffs[1][0];
  // if (fabs(det) < 1e-10) {
  //   return NPP_COEFFICIENT_ERROR;
  // }

  return NPP_SUCCESS;
}

NppStatus nppiWarpAffine_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffine_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffine_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffine_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation,
                                   nppStreamCtx);
}

NppStatus nppiWarpAffine_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                    int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffine_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffine_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffine_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation,
                                   nppStreamCtx);
}

NppStatus nppiWarpAffine_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffine_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                         eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffine_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3], int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffine_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs, eInterpolation,
                                    nppStreamCtx);
}

// ============================================================================
// nppiWarpPerspectiveBack API
// ============================================================================

extern "C" {

NppStatus nppiWarpAffineBack_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_8u_C4R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                             int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_16u_C1R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_16u_C3R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_16u_C4R_Ctx_impl(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32f_C3R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32f_C4R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32s_C1R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32s_C3R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiWarpAffineBack_32s_C4R_Ctx_impl(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                              Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                              int eInterpolation, NppStreamContext nppStreamCtx);
}

// ============================================================================
// WarpAffineBack API implement
// ============================================================================

// 8u C1R
NppStatus nppiWarpAffineBack_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                        int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                       eInterpolation, nppStreamCtx);
}

// 8u C3R
NppStatus nppiWarpAffineBack_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                        int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                       eInterpolation, nppStreamCtx);
}

// 8u C4R
NppStatus nppiWarpAffineBack_8u_C4R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                        Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                        int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_8u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                            eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_8u_C4R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                    Npp8u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                    int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_8u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                       eInterpolation, nppStreamCtx);
}

// 16u C1R
NppStatus nppiWarpAffineBack_16u_C1R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_16u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_16u_C1R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_16u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 16u C3R
NppStatus nppiWarpAffineBack_16u_C3R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_16u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_16u_C3R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_16u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 16u C4R
NppStatus nppiWarpAffineBack_16u_C4R_Ctx(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_16u_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_16u_C4R(const Npp16u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp16u *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_16u_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32f C1R
NppStatus nppiWarpAffineBack_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32f C3R
NppStatus nppiWarpAffineBack_32f_C3R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32f_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32f_C3R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32f_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32f C4R
NppStatus nppiWarpAffineBack_32f_C4R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32f_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32f_C4R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32f *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32f_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32s C1R
NppStatus nppiWarpAffineBack_32s_C1R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32s_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32s_C1R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32s_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32s C3R
NppStatus nppiWarpAffineBack_32s_C3R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32s_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32s_C3R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32s_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}

// 32s C4R
NppStatus nppiWarpAffineBack_32s_C4R_Ctx(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                         Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                         int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateWarpAffineInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                              eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiWarpAffineBack_32s_C4R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                             eInterpolation, nppStreamCtx);
}

NppStatus nppiWarpAffineBack_32s_C4R(const Npp32s *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                     Npp32s *pDst, int nDstStep, NppiRect oDstROI, const double aCoeffs[2][3],
                                     int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiWarpAffineBack_32s_C4R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, aCoeffs,
                                        eInterpolation, nppStreamCtx);
}