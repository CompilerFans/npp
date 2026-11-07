#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiYUV422ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU, int nSrcUStep,
                                            const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUV422ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU, int nSrcUStep,
                                          const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDstR, Npp8u *pDstG,
                                          Npp8u *pDstB, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUV422ToRGB_8u_P3AC4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcU, int nSrcUStep,
                                             const Npp8u *pSrcV, int nSrcVStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUV422ToRGB_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, cudaStream_t stream);
}

// Measured reference extent rule for the 422 inverse: the full width is
// converted but only height & ~1 rows; an odd height returns
// NPP_DOUBLE_SIZE_WARNING.
static inline NppiSize yuv422InvRoi(NppiSize roi) { roi.height &= ~1; return roi; }
static inline NppStatus yuv422InvStatus(cudaError_t s, NppiSize roi) {
  if (s != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return (roi.height & 1) ? NPP_DOUBLE_SIZE_WARNING: NPP_NO_ERROR;
}

static inline NppStatus validateYUV422ToRGBInputs(const Npp8u *const pSrc[3], const int rSrcStep[3], void *pDst,
                                                  int nDstStep, NppiSize oSizeROI, int dstChannels) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep) {
    return NPP_NULL_POINTER_ERROR;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  const int minDstStep = dstChannels == 0 ? oSizeROI.width: oSizeROI.width * dstChannels;
  if (nDstStep < minDstStep) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePackedYUV422ToRGB(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                  NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (nDstStep < oSizeROI.width * 3) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiYUV422ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYUV422ToRGBInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv422InvRoi(oSizeROI);
  cudaError_t cudaStatus = nppiYUV422ToRGB_8u_P3C3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                           rSrcStep[2], pDst, nDstStep, roi, nppStreamCtx.hStream);
  return yuv422InvStatus(cudaStatus, oSizeROI);
}

NppStatus nppiYUV422ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiYUV422ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV422ToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  NppStatus status = validateYUV422ToRGBInputs(pSrc, rSrcStep, pDst[0], nDstStep, oSizeROI, 0);
  if (status != NPP_SUCCESS) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv422InvRoi(oSizeROI);
  cudaError_t cudaStatus = nppiYUV422ToRGB_8u_P3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                         rSrcStep[2], pDst[0], pDst[1], pDst[2], nDstStep, roi,
                                                         nppStreamCtx.hStream);
  return yuv422InvStatus(cudaStatus, oSizeROI);
}

NppStatus nppiYUV422ToRGB_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiYUV422ToRGB_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV422ToRGB_8u_P3AC4R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYUV422ToRGBInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv422InvRoi(oSizeROI);
  cudaError_t cudaStatus = nppiYUV422ToRGB_8u_P3AC4R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                            rSrcStep[2], pDst, nDstStep, roi,
                                                            nppStreamCtx.hStream);
  return yuv422InvStatus(cudaStatus, oSizeROI);
}

NppStatus nppiYUV422ToRGB_8u_P3AC4R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiYUV422ToRGB_8u_P3AC4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedYUV422ToRGB(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  cudaError_t cudaStatus = nppiYUV422ToRGB_8u_C2C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                           nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR: NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUV422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiYUV422ToRGB_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
