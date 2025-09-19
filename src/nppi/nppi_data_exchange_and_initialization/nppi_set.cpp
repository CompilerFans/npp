#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Set Functions Implementation
 * Implements nppiSet functions for image data initialization
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiSet_8u_C1R_Ctx_cuda(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
NppStatus nppiSet_8u_C3R_Ctx_cuda(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx);
NppStatus nppiSet_32f_C1R_Ctx_cuda(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateSetInputs(void *pDst, int nDstStep, NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

/**
 * Set single channel 8-bit unsigned image to constant value
 */
NppStatus nppiSet_8u_C1R_Ctx(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  NppStatus status = validateSetInputs(pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSet_8u_C1R_Ctx_cuda(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C1R(Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSet_8u_C1R_Ctx(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * Set three channel 8-bit unsigned image to constant values
 */
NppStatus nppiSet_8u_C3R_Ctx(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                             NppStreamContext nppStreamCtx) {
  NppStatus status = validateSetInputs(pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  if (!aValue) {
    return NPP_NULL_POINTER_ERROR;
  }

  return nppiSet_8u_C3R_Ctx_cuda(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_8u_C3R(const Npp8u aValue[3], Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSet_8u_C3R_Ctx(aValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * Set single channel 32-bit float image to constant value
 */
NppStatus nppiSet_32f_C1R_Ctx(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                              NppStreamContext nppStreamCtx) {
  NppStatus status = validateSetInputs(pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiSet_32f_C1R_Ctx_cuda(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiSet_32f_C1R(Npp32f nValue, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSet_32f_C1R_Ctx(nValue, pDst, nDstStep, oSizeROI, nppStreamCtx);
}