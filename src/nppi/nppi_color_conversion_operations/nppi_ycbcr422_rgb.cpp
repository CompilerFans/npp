#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr422_8u_AC4C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr422_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr422_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                              Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                              NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiYCbCr422ToRGB_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToBGR_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                              const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToBGR_8u_C2C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream);
cudaError_t nppiYCbCr422_8u_C2P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                         Npp8u *pDstCb, int nDstCbStep, Npp8u *pDstCr, int nDstCrStep,
                                         NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422_8u_P3C2R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcCb, int nSrcCbStep,
                                         const Npp8u *pSrcCr, int nSrcCrStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToYCrCb422_8u_C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                                 NppiSize oSizeROI, cudaStream_t stream);
}

static inline NppStatus validateYCbCr422C2Inputs(const void *pSrc, int nSrcStep, void *pDst, int nDstStep,
                                                 NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validateYCbCr422ToPlanarInputs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                                       NppiSize oSizeROI, int srcChannels) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rDstStep) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (nSrcStep < oSizeROI.width * srcChannels || rDstStep[0] < oSizeROI.width || rDstStep[1] < oSizeROI.width / 2 ||
      rDstStep[2] < oSizeROI.width / 2) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validateYCbCr422FromPlanarInputs(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst,
                                                         int nDstStep, NppiSize oSizeROI, int dstChannels) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rSrcStep) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (rSrcStep[0] < oSizeROI.width || rSrcStep[1] < oSizeROI.width / 2 || rSrcStep[2] < oSizeROI.width / 2 ||
      nDstStep < oSizeROI.width * dstChannels) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

static inline NppStatus validateYCbCr422PackedLayoutInputs(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst,
                                                           int nDstStep, NppiSize oSizeROI) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0) {
    return NPP_SIZE_ERROR;
  }
  if ((oSizeROI.width % 2) != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 2 || nDstStep < oSizeROI.width * 2) {
    return NPP_STEP_ERROR;
  }
  return NPP_SUCCESS;
}

NppStatus nppiRGBToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYCbCr422_8u_C3C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYCbCr422ToRGB_8u_C2C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToYCbCr422_8u_C3C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToBGR_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYCbCr422ToBGR_8u_C2C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToBGR_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToBGR_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr422_8u_AC4C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToYCbCr422_8u_AC4C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr422_8u_AC4C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422ToPlanarInputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr422_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1],
                                                             pDst[2], rDstStep[2], oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr422_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422ToPlanarInputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr422_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1],
                                                             pDst[2], rDstStep[2], oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr422_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422FromPlanarInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr422ToRGB_8u_P3C3R_kernel(
      pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2], rSrcStep[2], pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToRGB_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422FromPlanarInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr422ToBGR_8u_P3C3R_kernel(
      pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2], rSrcStep[2], pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToBGR_8u_P3C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToBGR_8u_C2C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                         Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYCbCr422ToBGR_8u_C2C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToBGR_8u_C2C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}

NppStatus nppiYCbCr422_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  const NppStatus status = validateYCbCr422ToPlanarInputs(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, 2);
  if (status != NPP_SUCCESS) {
    return status;
  }

  const cudaError_t cudaStatus = nppiYCbCr422_8u_C2P3R_kernel(
      pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2], rDstStep[2], oSizeROI,
      nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422_8u_C2P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                NppiSize oSizeROI) {
  NppStreamContext context{};
  const NppStatus status = nppGetStreamContext(&context);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr422_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, context);
}

NppStatus nppiYCbCr422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  const NppStatus status = validateYCbCr422FromPlanarInputs(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, 2);
  if (status != NPP_SUCCESS) {
    return status;
  }

  const cudaError_t cudaStatus = nppiYCbCr422_8u_P3C2R_kernel(pSrc[0], rSrcStep[0], pSrc[1], rSrcStep[1], pSrc[2],
                                                              rSrcStep[2], pDst, nDstStep, oSizeROI,
                                                              nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422_8u_P3C2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext context{};
  const NppStatus status = nppGetStreamContext(&context);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, context);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  const NppStatus status = validateYCbCr422PackedLayoutInputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  const cudaError_t cudaStatus =
      nppiYCbCr422ToYCrCb422_8u_C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return cudaStatus == cudaSuccess ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToYCrCb422_8u_C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI) {
  NppStreamContext context{};
  const NppStatus status = nppGetStreamContext(&context);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, context);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst,
                                               int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !rSrcStep) {
    return nppiYCbCr422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
  }
  const Npp8u *swappedSrc[3] = {pSrc[0], pSrc[2], pSrc[1]};
  int swappedSteps[3] = {rSrcStep[0], rSrcStep[2], rSrcStep[1]};
  return nppiYCbCr422_8u_P3C2R_Ctx(swappedSrc, swappedSteps, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R(const Npp8u *const pSrc[3], int rSrcStep[3], Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI) {
  NppStreamContext context{};
  const NppStatus status = nppGetStreamContext(&context);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, context);
}

NppStatus nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                               NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pDst || !rDstStep) {
    return nppiYCbCr422_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
  }
  Npp8u *swappedDst[3] = {pDst[0], pDst[2], pDst[1]};
  int swappedSteps[3] = {rDstStep[0], rDstStep[2], rDstStep[1]};
  return nppiYCbCr422_8u_C2P3R_Ctx(pSrc, nSrcStep, swappedDst, swappedSteps, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCrCb422ToYCbCr422_8u_C2P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                           NppiSize oSizeROI) {
  NppStreamContext context{};
  const NppStatus status = nppGetStreamContext(&context);
  if (status != NPP_SUCCESS) {
    return status;
  }
  return nppiYCrCb422ToYCbCr422_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, context);
}
