#include "npp.h"
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiRotate_8u_C1R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                     int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus nppiRotate_8u_C3R_Ctx_impl(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                     int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                     int eInterpolation, NppStreamContext nppStreamCtx);

NppStatus nppiRotate_32f_C1R_Ctx_impl(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                      Npp32f *pDst, int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX,
                                      double nShiftY, int eInterpolation, NppStreamContext nppStreamCtx);
}

static inline NppStatus validateRotateInputs(const void *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI,
                                             void *pDst, int nDstStep, NppiRect oDstROI) {
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nSrcStep <= 0 || nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0)
    return NPP_SIZE_ERROR;
  if (oSrcROI.width <= 0 || oSrcROI.height <= 0)
    return NPP_SIZE_ERROR;
  if (oDstROI.width <= 0 || oDstROI.height <= 0)
    return NPP_SIZE_ERROR;

  // Check ROI bounds
  if (oSrcROI.x < 0 || oSrcROI.y < 0 || oSrcROI.x + oSrcROI.width > oSrcSize.width ||
      oSrcROI.y + oSrcROI.height > oSrcSize.height) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiRotate_8u_C1R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                int eInterpolation, NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateRotateInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRotate_8u_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX,
                                    nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C1R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                            int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                            int eInterpolation) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiRotate_8u_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY,
                               eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C3R_Ctx(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                                int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                int eInterpolation, NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateRotateInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRotate_8u_C3R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX,
                                    nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_8u_C3R(const Npp8u *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp8u *pDst,
                            int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                            int eInterpolation) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiRotate_8u_C3R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY,
                               eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_C1R_Ctx(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                                 int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateRotateInputs(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRotate_32f_C1R_Ctx_impl(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX,
                                     nShiftY, eInterpolation, nppStreamCtx);
}

NppStatus nppiRotate_32f_C1R(const Npp32f *pSrc, NppiSize oSrcSize, int nSrcStep, NppiRect oSrcROI, Npp32f *pDst,
                             int nDstStep, NppiRect oDstROI, double nAngle, double nShiftX, double nShiftY,
                             int eInterpolation) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiRotate_32f_C1R_Ctx(pSrc, oSrcSize, nSrcStep, oSrcROI, pDst, nDstStep, oDstROI, nAngle, nShiftX, nShiftY,
                                eInterpolation, nppStreamCtx);
}
