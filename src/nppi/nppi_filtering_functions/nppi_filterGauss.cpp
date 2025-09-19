#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Gaussian Filter Functions Implementation
 * Implements nppiFilterGauss functions for Gaussian blur filtering
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiFilterGauss_8u_C1R_Ctx_cuda_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                NppStreamContext nppStreamCtx);
NppStatus nppiFilterGauss_8u_C3R_Ctx_cuda_fixed(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                NppStreamContext nppStreamCtx);
NppStatus nppiFilterGauss_32f_C1R_Ctx_cuda_fixed(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                 NppiSize oSizeROI, NppiMaskSize eMaskSize,
                                                 NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateFilterGaussInputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                  NppiSize oSizeROI, NppiMaskSize eMaskSize) {
  // NVIDIA NPP behavior: zero-size ROI returns success (no processing needed)
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  // Early return for zero-size ROI (NVIDIA NPP compatible behavior)
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  // Validate mask size - only common sizes supported
  if (eMaskSize != NPP_MASK_SIZE_3_X_3 && eMaskSize != NPP_MASK_SIZE_5_X_5 && eMaskSize != NPP_MASK_SIZE_7_X_7) {
    return NPP_NOT_SUPPORTED_MODE_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel Gaussian filter
NppStatus nppiFilterGauss_8u_C1R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterGaussInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterGauss_8u_C1R_Ctx_cuda_fixed(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterGauss_8u_C1R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterGauss_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

// 8-bit unsigned three channel Gaussian filter
NppStatus nppiFilterGauss_8u_C3R_Ctx(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep,
                                     NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterGaussInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterGauss_8u_C3R_Ctx_cuda_fixed(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterGauss_8u_C3R(const Npp8u *pSrc, Npp32s nSrcStep, Npp8u *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                 NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterGauss_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

// 32-bit float single channel Gaussian filter
NppStatus nppiFilterGauss_32f_C1R_Ctx(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep,
                                      NppiSize oSizeROI, NppiMaskSize eMaskSize, NppStreamContext nppStreamCtx) {
  NppStatus status = validateFilterGaussInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiFilterGauss_32f_C1R_Ctx_cuda_fixed(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}

NppStatus nppiFilterGauss_32f_C1R(const Npp32f *pSrc, Npp32s nSrcStep, Npp32f *pDst, Npp32s nDstStep, NppiSize oSizeROI,
                                  NppiMaskSize eMaskSize) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiFilterGauss_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eMaskSize, nppStreamCtx);
}