#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
NppStatus nppiSwapChannels_8u_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);
NppStatus nppiSwapChannels_8u_C4R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, const int aDstOrder[4], NppStreamContext nppStreamCtx);
NppStatus nppiSwapChannels_32f_C3R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, const int aDstOrder[3], NppStreamContext nppStreamCtx);
}

static inline NppStatus validateSwapChannelsInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                   NppiSize oSizeROI, const int *aDstOrder) {
  if (!pSrc || !pDst || !aDstOrder) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiSwapChannels_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const int aDstOrder[3], NppStreamContext nppStreamCtx) {
  NppStatus status = validateSwapChannelsInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiSwapChannels_8u_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const int aDstOrder[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSwapChannels_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const int aDstOrder[4], NppStreamContext nppStreamCtx) {
  NppStatus status = validateSwapChannelsInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiSwapChannels_8u_C4R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_8u_C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  const int aDstOrder[4]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSwapChannels_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                       const int aDstOrder[3], NppStreamContext nppStreamCtx) {
  NppStatus status = validateSwapChannelsInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiSwapChannels_32f_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}

NppStatus nppiSwapChannels_32f_C3R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                   const int aDstOrder[3]) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiSwapChannels_32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aDstOrder, nppStreamCtx);
}
