#include "../../npp_internal.h"
#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

extern "C" {
NppStatus sumWindowRow_8u32f_Impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                  Npp32s nMaskSize, Npp32s nAnchor, cudaStream_t hStream);

NppStatus sumWindowColumn_8u32f_Impl(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor, cudaStream_t hStream);
}

// validate
static inline NppStatus validateSumWindowInputs(const void *pSrc, Npp32s nSrcStep, void *pDst, Npp32s nDstStep,
                                                NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor) {

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oROI.width <= 0 || oROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nMaskSize <= 0) {
    return NPP_MASK_SIZE_ERROR;
  }

  if (nAnchor < 0 || nAnchor >= nMaskSize) {
    return NPP_ANCHOR_ERROR;
  }

  int minSrcStep = oROI.width * sizeof(Npp8u);
  if (nSrcStep < minSrcStep) {
    return NPP_STEP_ERROR;
  }

  int minDstStep = oROI.width * sizeof(Npp32f);
  if (nDstStep < minDstStep) {
    return NPP_STEP_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiSumWindowRow_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                         NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor,
                                         NppStreamContext nppStreamCtx) {

  NppStatus status = validateSumWindowInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor);

  if (status != NPP_SUCCESS) {
    return status;
  }

  return sumWindowRow_8u32f_Impl(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx.hStream);
}

NppStatus nppiSumWindowRow_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oROI,
                                     Npp32s nMaskSize, Npp32s nAnchor) {

  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  nppStreamCtx.hStream = 0; // default stream

  return nppiSumWindowRow_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx);
}

NppStatus nppiSumWindowColumn_8u32f_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                            NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor,
                                            NppStreamContext nppStreamCtx) {

  NppStatus status = validateSumWindowInputs(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor);

  if (status != NPP_SUCCESS) {
    return status;
  }

  return sumWindowColumn_8u32f_Impl(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx.hStream);
}

NppStatus nppiSumWindowColumn_8u32f_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                        NppiSize oROI, Npp32s nMaskSize, Npp32s nAnchor) {

  NppStreamContext nppStreamCtx;
  memset(&nppStreamCtx, 0, sizeof(NppStreamContext));
  nppStreamCtx.hStream = 0;

  return nppiSumWindowColumn_8u32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oROI, nMaskSize, nAnchor, nppStreamCtx);
}