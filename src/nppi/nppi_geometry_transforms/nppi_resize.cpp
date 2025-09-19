#include "npp.h"
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

/**
 * NPP Image Resize Functions Implementation
 * Implements nppiResize functions for scaling images
 */

// Forward declarations for CUDA implementations
extern "C" {
NppStatus nppiResize_8u_C1R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_8u_C3R_Ctx_cuda(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                     Npp8u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                     int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_16u_C1R_Ctx_cuda(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_C1R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
NppStatus nppiResize_32f_C3R_Ctx_cuda(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                      Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                      int eInterpolation, NppStreamContext nppStreamCtx);
}

// Input validation helper
static inline NppStatus validateResizeInputs(const void *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                             void *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                             int eInterpolation) {
  if (oSrcSize.width <= 0 || oSrcSize.height <= 0 || oDstSize.width <= 0 || oDstSize.height <= 0) {
    return NPP_SIZE_ERROR;
  }

  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  // 验证插值参数以避免未使用警告
  if (eInterpolation < 0) {
    return NPP_BAD_ARGUMENT_ERROR;
  }

  // Basic ROI validation
  if (oSrcRectROI.x < 0 || oSrcRectROI.y < 0 || oSrcRectROI.x + oSrcRectROI.width > oSrcSize.width ||
      oSrcRectROI.y + oSrcRectROI.height > oSrcSize.height) {
    return NPP_SIZE_ERROR;
  }

  if (oDstRectROI.x < 0 || oDstRectROI.y < 0 || oDstRectROI.x + oDstRectROI.width > oDstSize.width ||
      oDstRectROI.y + oDstRectROI.height > oDstSize.height) {
    return NPP_SIZE_ERROR;
  }

  return NPP_SUCCESS;
}

// 8-bit unsigned single channel resize
NppStatus nppiResize_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_C1R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 8-bit unsigned three channel resize
NppStatus nppiResize_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                                int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation,
                                NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_8u_C3R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                    eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_8u_C3R(const Npp8u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u *pDst,
                            int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                               eInterpolation, nppStreamCtx);
}

// 16-bit unsigned single channel resize
NppStatus nppiResize_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp16u *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_16u_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_16u_C1R(const Npp16u *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16u *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float single channel resize
NppStatus nppiResize_32f_C1R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_C1R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_C1R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}

// 32-bit float three channel resize
NppStatus nppiResize_32f_C3R_Ctx(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI,
                                 Npp32f *pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI,
                                 int eInterpolation, NppStreamContext nppStreamCtx) {
  NppStatus status = validateResizeInputs(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                          eInterpolation);
  if (status != NPP_SUCCESS) {
    return status;
  }

  return nppiResize_32f_C3R_Ctx_cuda(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                     eInterpolation, nppStreamCtx);
}

NppStatus nppiResize_32f_C3R(const Npp32f *pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f *pDst,
                             int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation) {
  NppStreamContext nppStreamCtx;
  nppStreamCtx.hStream = 0;
  return nppiResize_32f_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI,
                                eInterpolation, nppStreamCtx);
}