#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYCbCr411_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr411_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr411_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr411_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr411_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                            Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                            int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr411_8u_P3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                            Npp8u *pDstY, int nDstYStep, Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr,
                                            int nDstCrStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr411ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr411ToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr411ToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream);
cudaError_t nppiYCbCr411ToBGR_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream);
cudaError_t nppiYCbCr411ToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                            const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDstR, Npp8u *pDstG,
                                            Npp8u *pDstB, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr411ToBGR_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                            const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDstB, Npp8u *pDstG,
                                            Npp8u *pDstR, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
}

static inline NppStatus validatePackedToPlanar411(const Npp8u *pSrc, int nSrcStep, Npp8u *const pDst[3],
                                                  const int rDstStep[3], NppiSize oSizeROI, int srcChannels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || (oSizeROI.width % 4) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * srcChannels) {
    return NPP_STEP_ERROR;
  }
  if (rDstStep[0] < oSizeROI.width || rDstStep[1] < oSizeROI.width / 4 || rDstStep[2] < oSizeROI.width / 4) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanarToPlanar411(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3],
                                                  int nDstStep, NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || (oSizeROI.width % 4) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanar411ToPacked(const Npp8u *const pSrc[3], const int rSrcStep[3], Npp8u *pDst,
                                                  int nDstStep, NppiSize oSizeROI, int dstChannels) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!pSrc[0] || !pSrc[1] || !pSrc[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || (oSizeROI.width % 4) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (rSrcStep[0] < oSizeROI.width || rSrcStep[1] < oSizeROI.width / 4 || rSrcStep[2] < oSizeROI.width / 4) {
    return NPP_STEP_ERROR;
  }
  if (nDstStep < oSizeROI.width * dstChannels) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validatePlanar411ToPlanar(const Npp8u *const pSrc[3], const int rSrcStep[3], Npp8u *pDst[3],
                                                  int nDstStep, NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || (oSizeROI.width % 4) != 0) {
    return NPP_SIZE_ERROR;
  }
  if (rSrcStep[0] < oSizeROI.width || rSrcStep[1] < oSizeROI.width / 4 || rSrcStep[2] < oSizeROI.width / 4) {
    return NPP_STEP_ERROR;
  }
  if (nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiRGBToYCbCr411_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanar411(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr411_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                             rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr411_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr411_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr411_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanar411(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr411_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                             rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr411_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr411_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr411_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanar411(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr411_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                              rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                              nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr411_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr411_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr411_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanar411(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr411_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                              rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                              nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr411_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr411_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr411_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPlanar(pSrc, rDstStep, pDst, nSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr411_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], rDstStep[0],
                                                           pDst[1], rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                           nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr411_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr411_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr411_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPlanar(pSrc, rDstStep, pDst, nSrcStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr411_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], rDstStep[0],
                                                           pDst[1], rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                           nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr411_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr411_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr411ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPacked(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToRGB_8u_P3C3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                             rSrcStep[2], pDst, nDstStep, oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr411ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPacked(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToBGR_8u_P3C3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                             rSrcStep[2], pDst, nDstStep, oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToBGR_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr411ToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPacked(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToRGB_8u_P3C4R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                             rSrcStep[2], pDst, nDstStep, oSizeROI, nAval,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToRGB_8u_P3C4R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToRGB_8u_P3C4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}

NppStatus nppiYCbCr411ToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPacked(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToBGR_8u_P3C4R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                             rSrcStep[2], pDst, nDstStep, oSizeROI, nAval,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToBGR_8u_P3C4R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToBGR_8u_P3C4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}

NppStatus nppiYCbCr411ToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPlanar(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToRGB_8u_P3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                           rSrcStep[2], pDst[0], pDst[1], pDst[2], nDstStep,
                                                           oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToRGB_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToRGB_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr411ToBGR_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanar411ToPlanar(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr411ToBGR_8u_P3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                           rSrcStep[2], pDst[0], pDst[1], pDst[2], nDstStep,
                                                           oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr411ToBGR_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr411ToBGR_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

// JPEG variants: reuse base implementations.
NppStatus nppiRGBToYCbCr411_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiRGBToYCbCr411_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToYCbCr411_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI) {
  return nppiRGBToYCbCr411_8u_C3P3R(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
}

NppStatus nppiBGRToYCbCr411_JPEG_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiBGRToYCbCr411_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr411_JPEG_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI) {
  return nppiBGRToYCbCr411_8u_C3P3R(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
}

NppStatus nppiRGBToYCbCr411_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiRGBToYCbCr411_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToYCbCr411_JPEG_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI) {
  return nppiRGBToYCbCr411_8u_P3R(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
}

NppStatus nppiBGRToYCbCr411_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiBGRToYCbCr411_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCbCr411_JPEG_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI) {
  return nppiBGRToYCbCr411_8u_P3R(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
}

NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiYCbCr411ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI) {
  return nppiYCbCr411ToRGB_8u_P3C3R(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
}

NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiYCbCr411ToBGR_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI) {
  return nppiYCbCr411ToBGR_8u_P3C3R(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
}

NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiYCbCr411ToRGB_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr411ToRGB_JPEG_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI) {
  return nppiYCbCr411ToRGB_8u_P3R(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
}

NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiYCbCr411ToBGR_8u_P3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr411ToBGR_JPEG_8u_P3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI) {
  return nppiYCbCr411ToBGR_8u_P3R(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
}
