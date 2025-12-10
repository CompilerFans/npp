#include "../../npp_version_compat.h"
#include "npp.h"
#include <cuda_runtime.h>

// Forward declaration for CUDA implementation
extern "C" {
NppStatus nppiRectStdDev_32s32f_C1R_Ctx_impl(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep,
                                              Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect,
                                              NppStreamContext nppStreamCtx);
}

//=============================================================================
// nppiRectStdDev_32s32f_C1R - 32-bit signed to 32-bit float RectStdDev
//=============================================================================

NppStatus nppiRectStdDev_32s32f_C1R_Ctx(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep,
                                         Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect,
                                         NppStreamContext nppStreamCtx) {
  // Validate input parameters
  if (!pSrc || !pSqr || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nSqrStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (oRect.width <= 0 || oRect.height <= 0) {
    return NPP_RECTANGLE_ERROR;
  }

  if (oRect.x < 0 || oRect.y < 0) {
    return NPP_RECTANGLE_ERROR;
  }

  // Call CUDA implementation
  return nppiRectStdDev_32s32f_C1R_Ctx_impl(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect,
                                             nppStreamCtx);
}

NppStatus nppiRectStdDev_32s32f_C1R(const Npp32s *pSrc, int nSrcStep, const Npp64f *pSqr, int nSqrStep,
                                     Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppiRect oRect) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiRectStdDev_32s32f_C1R_Ctx(pSrc, nSrcStep, pSqr, nSqrStep, pDst, nDstStep, oSizeROI, oRect, nppStreamCtx);
}
