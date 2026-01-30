#include "npp.h"
#include "nppi_color_conversion_validation.hpp"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations

extern "C" {
cudaError_t nppiRGBToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                          Npp8u *pDstV, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiBGRToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiBGRToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiBGRToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYUV_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiBGRToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU,
                                          Npp8u *pDstV, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiYUVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiYUVToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUVToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUVToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                       Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiYUVToBGR_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiYUVToBGR_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUVToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUVToBGR_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcU, const Npp8u *pSrcV,
                                       Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToYCbCr_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                           Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYCbCr_8u_P3R_kernel(const Npp8u *pSrcR, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcB,
                                         Npp8u *pDstY, Npp8u *pDstCb, Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI,
                                         cudaStream_t stream);
cudaError_t nppiRGBToYCbCr_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                           Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToYCbCr_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstCb,
                                            Npp8u *pDstCr, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                            cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_P3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                         const Npp8u *pSrcCr, Npp8u *pDstR, Npp8u *pDstG, Npp8u *pDstB,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYCbCrToRGB_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           Npp8u alpha, cudaStream_t stream);
cudaError_t nppiYCbCrToBGR_8u_P3C3R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           cudaStream_t stream);
cudaError_t nppiYCbCrToBGR_8u_P3C4R_kernel(const Npp8u *pSrcY, int nSrcStep, const Npp8u *pSrcCb,
                                           const Npp8u *pSrcCr, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                           Npp8u alpha, cudaStream_t stream);
}

NppStatus nppiRGBToYUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiRGBToYUV_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiRGBToYUV_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYUV_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYUV_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYUV_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2],
                                                      nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 4, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYUV_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3],
                                                        nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYUV_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYUV_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToYUV_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  const Npp8u *srcPlanes[3] = {pSrc[2], pSrc[1], pSrc[0]};
  NppStatus status = validatePlanarInput(srcPlanes, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYUV_8u_P3R_kernel(srcPlanes[0], nSrcStep, srcPlanes[1], srcPlanes[2], pDst[0],
                                                      pDst[1], pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 4, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYUV_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3],
                                                        nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiYUVToRGB_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiYUVToRGB_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYUVToRGB_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToRGB_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYUVToRGB_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToRGB_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYUVToRGB_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2],
                                                      nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToRGB_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToRGB_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToBGR_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYUVToBGR_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToBGR_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToBGR_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToBGR_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYUVToBGR_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToBGR_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToBGR_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYUVToBGR_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToBGR_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToBGR_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYUVToBGR_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2],
                                                      nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYUVToBGR_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYUVToBGR_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYCbCr_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2],
                                                        nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYCbCr_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYCbCr_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToYCbCr_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep,
                                                          oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYCbCr_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYCbCr_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToYCbCr_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep,
                                                          oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYCbCr_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 4, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToYCbCr_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3],
                                                          nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYCbCr_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYCbCr_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToRGB_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToRGB_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                     NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiYCbCrToRGB_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToRGB_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToRGB_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToRGB_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep,
                                                         oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToRGB_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToRGB_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToRGB_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToRGB_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2],
                                                        nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToRGB_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToRGB_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToRGB_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToRGB_8u_P3C4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep,
                                                         oSizeROI, nAval, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToRGB_8u_P3C4R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToRGB_8u_P3C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}

NppStatus nppiYCbCrToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToBGR_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep,
                                                         oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToBGR_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYCbCrToBGR_8u_P3C4R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiYCbCrToBGR_8u_P3C4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep,
                                                         oSizeROI, nAval, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiYCbCrToBGR_8u_P3C4R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                  NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiYCbCrToBGR_8u_P3C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, ctx);
}
