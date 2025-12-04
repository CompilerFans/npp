#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations

extern "C" {
NppStatus nppiConvert_8u32f_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiConvert_8u32f_C3R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiConvert_8u16u_C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx);
NppStatus nppiConvert_32f8u_C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppRoundMode eRoundMode, NppStreamContext nppStreamCtx);
NppStatus nppiConvert_16u32f_C1R_Ctx_impl(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

static inline NppStatus validateConvertInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                              NppiSize oSizeROI) {
  if (!pSrc || !pDst)
    return NPP_NULL_POINTER_ERROR;
  if (nSrcStep <= 0 || nDstStep <= 0)
    return NPP_STEP_ERROR;
  if (oSizeROI.width < 0 || oSizeROI.height < 0)
    return NPP_SIZE_ERROR;
  return NPP_SUCCESS;
}

NppStatus nppiConvert_8u32f_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  // Parameter validation
  NppStatus status = validateConvertInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiConvert_8u32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_8u32f_C1R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiConvert_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_8u32f_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  return nppiConvert_8u32f_C3R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_8u32f_C3R(const Npp8u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  // Get default stream context
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);

  return nppiConvert_8u32f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_8u16u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validateConvertInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiConvert_8u16u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_8u16u_C1R(const Npp8u *pSrc, int nSrcStep, Npp16u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiConvert_8u16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_32f8u_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppRoundMode eRoundMode, NppStreamContext nppStreamCtx) {
  NppStatus status = validateConvertInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiConvert_32f8u_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eRoundMode, nppStreamCtx);
}

NppStatus nppiConvert_32f8u_C1R(const Npp32f *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                NppRoundMode eRoundMode) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiConvert_32f8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eRoundMode, nppStreamCtx);
}

NppStatus nppiConvert_16u32f_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateConvertInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiConvert_16u32f_C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiConvert_16u32f_C1R(const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiConvert_16u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
