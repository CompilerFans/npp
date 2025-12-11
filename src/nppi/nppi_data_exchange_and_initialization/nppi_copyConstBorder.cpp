#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for implementation functions
extern "C" {
NppStatus nppiCopyConstBorder_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep,
                                               NppiSize oSrcSizeROI,
                                               Npp8u *pDst, int nDstStep,
                                               NppiSize oDstSizeROI,
                                               int nTopBorderHeight, int nLeftBorderWidth,
                                               Npp8u nValue,
                                               NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateCopyConstBorderInputs(const void *pSrc, int nSrcStep,
                                                       NppiSize oSrcSizeROI,
                                                       const void *pDst, int nDstStep,
                                                       NppiSize oDstSizeROI) {
  if (oSrcSizeROI.width <= 0 || oSrcSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (oDstSizeROI.width <= 0 || oDstSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

// ============================================================================
// nppiCopyConstBorder_8u_C1R - Copy with constant border
// ============================================================================

NppStatus nppiCopyConstBorder_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                          NppiSize oSrcSizeROI,
                                          Npp8u *pDst, int nDstStep,
                                          NppiSize oDstSizeROI,
                                          int nTopBorderHeight, int nLeftBorderWidth,
                                          Npp8u nValue,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateCopyConstBorderInputs(pSrc, nSrcStep, oSrcSizeROI,
                                                     pDst, nDstStep, oDstSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiCopyConstBorder_8u_C1R_Ctx_impl(pSrc, nSrcStep, oSrcSizeROI,
                                              pDst, nDstStep, oDstSizeROI,
                                              nTopBorderHeight, nLeftBorderWidth,
                                              nValue, nppStreamCtx);
}

NppStatus nppiCopyConstBorder_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                      NppiSize oSrcSizeROI,
                                      Npp8u *pDst, int nDstStep,
                                      NppiSize oDstSizeROI,
                                      int nTopBorderHeight, int nLeftBorderWidth,
                                      Npp8u nValue) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiCopyConstBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSizeROI,
                                         pDst, nDstStep, oDstSizeROI,
                                         nTopBorderHeight, nLeftBorderWidth,
                                         nValue, nppStreamCtx);
}
