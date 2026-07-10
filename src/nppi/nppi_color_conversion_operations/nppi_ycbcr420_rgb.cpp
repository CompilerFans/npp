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

static NppStatus validateYCbCr420Inputs(const Npp8u *pSrc, int nSrcStep, const Npp8u *const pDst[3],
                                        const int rDstStep[3], NppiSize oSizeROI, int srcChannels) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rDstStep) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width % 2) != 0 || (oSizeROI.height % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (nSrcStep < oSizeROI.width * srcChannels || rDstStep[0] < oSizeROI.width ||
      rDstStep[1] < oSizeROI.width / 2 || rDstStep[2] < oSizeROI.width / 2) {
    return NPP_STEP_ERROR;
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
  NppStatus status = validateYCbCr420Inputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
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
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiRGBToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr420Inputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
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
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr420Inputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 4);
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
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiBGRToYCbCr420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCrCb420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pDst || !rDstStep) {
    return nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
  }
  Npp8u *ycbcrDst[3] = {pDst[0], pDst[2], pDst[1]};
  int ycbcrSteps[3] = {rDstStep[0], rDstStep[2], rDstStep[1]};
  return nppiBGRToYCbCr420_8u_C3P3R_Ctx(pSrc, nSrcStep, ycbcrDst, ycbcrSteps, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCrCb420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiBGRToYCrCb420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCrCb420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pDst || !rDstStep) {
    return nppiBGRToYCbCr420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
  }
  Npp8u *ycbcrDst[3] = {pDst[0], pDst[2], pDst[1]};
  int ycbcrSteps[3] = {rDstStep[0], rDstStep[2], rDstStep[1]};
  return nppiBGRToYCbCr420_8u_AC4P3R_Ctx(pSrc, nSrcStep, ycbcrDst, ycbcrSteps, oSizeROI, nppStreamCtx);
}

NppStatus nppiBGRToYCrCb420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI) {
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiBGRToYCrCb420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                               int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3],
                                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pDst || !rDstStep) {
    return nppiYCbCr420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI,
                                     nppStreamCtx);
  }
  Npp8u *ycbcrDst[3] = {pDst[0], pDst[2], pDst[1]};
  int ycbcrSteps[3] = {rDstStep[0], rDstStep[2], rDstStep[1]};
  return nppiYCbCr420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, ycbcrDst, ycbcrSteps, oSizeROI,
                                   nppStreamCtx);
}

NppStatus nppiYCbCr420ToYCrCb420_8u_P2P3R(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCbCr,
                                           int nSrcCbCrStep, Npp8u *pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr420ToYCrCb420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep,
                                             oSizeROI, ctx);
}

NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                               int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep,
                                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !rSrcStep) {
    return nppiYCbCr420_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI,
                                     nppStreamCtx);
  }
  const Npp8u *ycbcrSrc[3] = {pSrc[0], pSrc[2], pSrc[1]};
  int ycbcrSteps[3] = {rSrcStep[0], rSrcStep[2], rSrcStep[1]};
  return nppiYCbCr420_8u_P3P2R_Ctx(ycbcrSrc, ycbcrSteps, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep, oSizeROI,
                                   nppStreamCtx);
}

NppStatus nppiYCrCb420ToYCbCr420_8u_P3P2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDstY,
                                           int nDstYStep, Npp8u *pDstCbCr, int nDstCbCrStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCrCb420ToYCbCr420_8u_P3P2R_Ctx(pSrc, rSrcStep, pDstY, nDstYStep, pDstCbCr, nDstCbCrStep,
                                             oSizeROI, ctx);
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
  NppStreamContext ctx{};
  const NppStatus status = nppGetStreamContext(&ctx);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr420ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
