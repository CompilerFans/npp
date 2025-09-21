#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>



// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiAbsDiff_8u_C1R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbsDiff_8u_C3R_Ctx_impl(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                      int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiAbsDiff_32f_C1R_Ctx_impl(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step,
                                       Npp32f *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateAbsDiffInputs(const void *pSrc1, int nSrc1Step, const void *pSrc2, int nSrc2Step,
                                              void *pDst, int nDstStep, NppiSize oSizeROI) {
  // Follow NPP standard error checking priority order
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrc1Step <= 0 || nSrc2Step <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc1 || !pSrc2 || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel absolute difference
NppStatus nppiAbsDiff_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiAbsDiff_8u_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiAbsDiff_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 8-bit unsigned three channel absolute difference
NppStatus nppiAbsDiff_8u_C3R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                 int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiAbsDiff_8u_C3R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_8u_C3R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                             int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiAbsDiff_8u_C3R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// 32-bit float single channel absolute difference
NppStatus nppiAbsDiff_32f_C1R_Ctx(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateAbsDiffInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiAbsDiff_32f_C1R_Ctx_impl(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiAbsDiff_32f_C1R(const Npp32f *pSrc1, int nSrc1Step, const Npp32f *pSrc2, int nSrc2Step, Npp32f *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiAbsDiff_32f_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
