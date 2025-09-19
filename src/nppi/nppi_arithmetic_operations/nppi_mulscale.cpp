#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP MulScale Operations Implementation
 * Multiplies two images and scales by maximum value for pixel bit width
 * For 8-bit: scale by 255, for 16-bit: scale by 65535
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiMulScale_8u_C1R_Ctx_cuda(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step,
                                       Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
NppStatus nppiMulScale_16u_C1R_Ctx_cuda(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step,
                                        Npp16u *pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx);
}

/**
 * Validate common input parameters for MulScale operations
 */
static inline NppStatus validateMulScaleInputs(const void *pSrc1, int nSrc1Step, const void *pSrc2, int nSrc2Step,
                                               void *pDst, int nDstStep, NppiSize oSizeROI) {
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

/**
 * Validate inputs for in-place MulScale operations
 */
static inline NppStatus validateMulScaleInPlaceInputs(const void *pSrc, int nSrcStep, void *pSrcDst, int nSrcDstStep,
                                                      NppiSize oSizeROI) {
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nSrcDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pSrcDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  return NPP_SUCCESS;
}

/**
 * 8-bit unsigned multiplication with scaling
 */
NppStatus nppiMulScale_8u_C1R_Ctx(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                                  int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMulScaleInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiMulScale_8u_C1R_Ctx_cuda(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1R(const Npp8u *pSrc1, int nSrc1Step, const Npp8u *pSrc2, int nSrc2Step, Npp8u *pDst,
                              int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiMulScale_8u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 8-bit unsigned multiplication with scaling - in place
 */
NppStatus nppiMulScale_8u_C1IR_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validateMulScaleInPlaceInputs(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiMulScale_8u_C1R_Ctx_cuda(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI,
                                      nppStreamCtx);
}

NppStatus nppiMulScale_8u_C1IR(const Npp8u *pSrc, int nSrcStep, Npp8u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiMulScale_8u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 16-bit unsigned multiplication with scaling
 */
NppStatus nppiMulScale_16u_C1R_Ctx(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                                   int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMulScaleInputs(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiMulScale_16u_C1R_Ctx_cuda(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1R(const Npp16u *pSrc1, int nSrc1Step, const Npp16u *pSrc2, int nSrc2Step, Npp16u *pDst,
                               int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiMulScale_16u_C1R_Ctx(pSrc1, nSrc1Step, pSrc2, nSrc2Step, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

/**
 * 16-bit unsigned multiplication with scaling - in place
 */
NppStatus nppiMulScale_16u_C1IR_Ctx(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateMulScaleInPlaceInputs(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiMulScale_16u_C1R_Ctx_cuda(pSrc, nSrcStep, pSrcDst, nSrcDstStep, pSrcDst, nSrcDstStep, oSizeROI,
                                       nppStreamCtx);
}

NppStatus nppiMulScale_16u_C1IR(const Npp16u *pSrc, int nSrcStep, Npp16u *pSrcDst, int nSrcDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiMulScale_16u_C1IR_Ctx(pSrc, nSrcStep, pSrcDst, nSrcDstStep, oSizeROI, nppStreamCtx);
}