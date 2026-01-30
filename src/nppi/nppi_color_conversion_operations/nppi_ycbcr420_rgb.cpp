#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYCbCr420_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr420_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr420_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                               Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                               NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiYCbCr420ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
}

static NppStatus validateYCbCr420Inputs(const Npp8u *const pDst[3], const int rDstStep[3], NppiSize oSizeROI) {
  if (!pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rDstStep || rDstStep[0] <= 0 || rDstStep[1] <= 0 || rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if ((oSizeROI.width % 2) != 0 || (oSizeROI.height % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_NO_ERROR;
}

static inline NppStatus validateYCbCr420ToRGBInputs(const Npp8u *const pSrc[3], const int rSrcStep[3], void *pDst,
                                                    int nDstStep, NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (rSrcStep[0] <= 0 || rSrcStep[1] <= 0 || rSrcStep[2] <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiRGBToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYCbCr420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr420_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                            rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYCbCr420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr420_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                            rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYCbCr420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr420_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1],
                                                             rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr420ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr420ToRGBInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr420ToRGB_8u_P3C3R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                            rSrcStep[2], pDst, nDstStep, oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr420ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
