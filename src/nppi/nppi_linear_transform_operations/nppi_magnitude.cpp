#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Magnitude Functions Implementation
 * Implements nppiMagnitude and nppiMagnitudeSqr functions for complex to real transformations
 */

// Forward declarations for mpp host func implementations
extern "C" {
NppStatus nppiMagnitude_32fc32f_C1R_Ctx_cuda(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx_cuda(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                                NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Helper function for parameter validation
 */
static NppStatus validateMagnitudeParameters(const void *pSrc, int nSrcStep, const void *pDst, int nDstStep,
                                             NppiSize oSizeROI, int srcElementSize, int dstElementSize) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  int minSrcStep = oSizeROI.width * srcElementSize;
  int minDstStep = oSizeROI.width * dstElementSize;

  if (nSrcStep < minSrcStep || nDstStep < minDstStep) {
    return NPP_STRIDE_ERROR;
  }

  return NPP_NO_ERROR;
}

// ============================================================================
// Magnitude functions
// ============================================================================

/**
 * 32-bit floating point complex to 32-bit floating point magnitude
 */
NppStatus nppiMagnitude_32fc32f_C1R_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateMagnitudeParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, sizeof(Npp32fc), sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiMagnitude_32fc32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMagnitude_32fc32f_C1R(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMagnitude_32fc32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// ============================================================================
// Magnitude squared functions
// ============================================================================

/**
 * 32-bit floating point complex to 32-bit floating point squared magnitude
 */
NppStatus nppiMagnitudeSqr_32fc32f_C1R_Ctx(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                           NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status =
      validateMagnitudeParameters(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, sizeof(Npp32fc), sizeof(Npp32f));
  if (status != NPP_NO_ERROR) {
    return status;
  }

  return nppiMagnitudeSqr_32fc32f_C1R_Ctx_cuda(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMagnitudeSqr_32fc32f_C1R(const Npp32fc *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep,
                                       NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppGetStreamContext(&nppStreamCtx);
  return nppiMagnitudeSqr_32fc32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

// ==================== 两通道浮点数幅度计算 (未实现) ====================

NppStatus nppiMagnitude_32f_C2R(const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI) {
  // 参数验证
  if (pSrc == nullptr || pDst == nullptr) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 2 * (int)sizeof(Npp32f) || nDstStep < oSizeROI.width * (int)sizeof(Npp32f)) {
    return NPP_STEP_ERROR;
  }

  // 打印未实现警告
  fprintf(stderr, "WARNING: nppiMagnitude_32f_C2R is not implemented in this NPP library build\n");

  return NPP_NOT_IMPLEMENTED_ERROR;
}