#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiYUV420ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                          int srcStepU, int srcStepV, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                          int dstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUV420ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                            int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                            cudaStream_t stream);
cudaError_t nppiYUV420ToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                            int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                            cudaStream_t stream);
cudaError_t nppiYUV420ToRGB_8u_P3AC4R_kernel(const Npp8u *pSrcY, const Npp8u *pSrcU, const Npp8u *pSrcV, int srcStepY,
                                             int srcStepU, int srcStepV, Npp8u *pDst, int dstStep, NppiSize oSizeROI,
                                             cudaStream_t stream);
}

static NppStatus validate_yuv420(const Npp8u *const pSrc[3], int rSrcStep[3], NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width & 1) || (oSizeROI.height & 1)) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (rSrcStep[0] < oSizeROI.width || rSrcStep[1] < (oSizeROI.width / 2) || rSrcStep[2] < (oSizeROI.width / 2)) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiYUV420ToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validate_yuv420(pSrc, rSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  if (!pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiYUV420ToRGB_8u_P3R_kernel(pSrc[0], pSrc[1], pSrc[2], rSrcStep[0], rSrcStep[1], rSrcStep[2], pDst[0], pDst[1],
                                    pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUV420ToRGB_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUV420ToRGB_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV420ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validate_yuv420(pSrc, rSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nDstStep < oSizeROI.width * 3) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiYUV420ToRGB_8u_P3C3R_kernel(pSrc[0], pSrc[1], pSrc[2], rSrcStep[0], rSrcStep[1], rSrcStep[2], pDst, nDstStep,
                                      oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUV420ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUV420ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV420ToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validate_yuv420(pSrc, rSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nDstStep < oSizeROI.width * 4) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiYUV420ToRGB_8u_P3C4R_kernel(pSrc[0], pSrc[1], pSrc[2], rSrcStep[0], rSrcStep[1], rSrcStep[2], pDst, nDstStep,
                                      oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUV420ToRGB_8u_P3C4R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUV420ToRGB_8u_P3C4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUV420ToRGB_8u_P3AC4R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validate_yuv420(pSrc, rSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }
  if (!pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nDstStep < oSizeROI.width * 4) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiYUV420ToRGB_8u_P3AC4R_kernel(pSrc[0], pSrc[1], pSrc[2], rSrcStep[0], rSrcStep[1], rSrcStep[2], pDst,
                                       nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUV420ToRGB_8u_P3AC4R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUV420ToRGB_8u_P3AC4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
