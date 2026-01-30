#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr422_8u_AC4C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiYCbCr422ToRGB_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToBGR_8u_C2C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCr422ToBGR_8u_C2C4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                              NppiSize oSizeROI, Npp8u alpha, cudaStream_t stream);
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

NppStatus nppiRGBToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr422_8u_C3C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToRGB_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr422ToRGB_8u_C2C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToRGB_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToRGB_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr422_8u_C3C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToBGR_8u_C2C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr422ToBGR_8u_C2C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToBGR_8u_C2C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToBGR_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr422_8u_AC4C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr422_8u_AC4C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                             nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr422_8u_AC4C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr422_8u_AC4C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCr422ToBGR_8u_C2C4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYCbCr422C2Inputs(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCr422ToBGR_8u_C2C4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval,
                                                            nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCr422ToBGR_8u_C2C4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCr422ToBGR_8u_C2C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}
