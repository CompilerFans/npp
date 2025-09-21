#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiRGBToGray_8u_C3C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx);
NppStatus nppiRGBToGray_8u_AC4C1R_Ctx_impl(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToGray_32f_C3C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiRGBToGray_32f_AC4C1R_Ctx_impl(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}
static inline NppStatus validateRGBToGrayInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
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
NppStatus nppiRGBToGray_8u_C3C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validateRGBToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRGBToGray_8u_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToGray_8u_C3C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRGBToGray_8u_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
NppStatus nppiRGBToGray_8u_AC4C1R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  NppStatus status = validateRGBToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRGBToGray_8u_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToGray_8u_AC4C1R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRGBToGray_8u_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
NppStatus nppiRGBToGray_32f_C3C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                      NppStreamContext nppStreamCtx) {
  NppStatus status = validateRGBToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRGBToGray_32f_C3C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToGray_32f_C3C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRGBToGray_32f_C3C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
NppStatus nppiRGBToGray_32f_AC4C1R_Ctx(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  NppStatus status = validateRGBToGrayInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiRGBToGray_32f_AC4C1R_Ctx_impl(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToGray_32f_AC4C1R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiRGBToGray_32f_AC4C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
