#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiColorTwist32f_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
NppStatus nppiColorTwist32f_8u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                            NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateColorTwistInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                 NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiColorTwist32f_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateColorTwistInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiColorTwist32f_8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiColorTwist32f_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}
