#include "npp.h"
#include "nppi_color_conversion_validation.hpp"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToHLS_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToHLS_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiHLSToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL,
                                         Npp8u *pDstS, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_P3C3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_P3R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                       Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstH, Npp8u *pDstL,
                                          Npp8u *pDstS, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_AP4C4R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                          const Npp8u *pSrcA, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiBGRToHLS_8u_AP4R_kernel(const Npp8u *pSrcB, int nSrcStep, const Npp8u *pSrcG, const Npp8u *pSrcR,
                                        const Npp8u *pSrcA, Npp8u *pDstH, Npp8u *pDstL, Npp8u *pDstS, Npp8u *pDstA,
                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                         Npp8u *pDstR, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_P3C3R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                         Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_P3R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                       Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstB, Npp8u *pDstG,
                                          Npp8u *pDstR, Npp8u *pDstA, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_AP4C4R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                          const Npp8u *pSrcA, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                          cudaStream_t stream);
cudaError_t nppiHLSToBGR_8u_AP4R_kernel(const Npp8u *pSrcH, int nSrcStep, const Npp8u *pSrcL, const Npp8u *pSrcS,
                                        const Npp8u *pSrcA, Npp8u *pDstB, Npp8u *pDstG, Npp8u *pDstR, Npp8u *pDstA,
                                        int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiHLSToRGB_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
}

NppStatus nppiRGBToHLS_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiRGBToHLS_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToHLS_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToHLS_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToHLS_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToHLS_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToHLS_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToHLS_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiHLSToRGB_8u_C3R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToRGB_8u_C3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToRGB_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToRGB_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiHLSToRGB_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToRGB_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToRGB_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToHLS_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToHLS_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI,
                                   nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToHLS_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep, oSizeROI,
                                   nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToHLS_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI,
                                 nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 4, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToHLS_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3],
                                                         nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_AP4C4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToHLS_8u_AP4C4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pSrc[3], pDst, nDstStep,
                                                         oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_AP4C4R(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_AP4C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToHLS_8u_AP4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiBGRToHLS_8u_AP4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pSrc[3], pDst[0], pDst[1],
                                                       pDst[2], pDst[3], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToHLS_8u_AP4R(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep,
                               NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToHLS_8u_AP4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 3, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiHLSToBGR_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI,
                                   nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                    NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiHLSToBGR_8u_P3C3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst, nDstStep, oSizeROI,
                                   nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_P3C3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst, int nDstStep,
                                NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_P3C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                  NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 3);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiHLSToBGR_8u_P3R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI,
                                 nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePackedToPlanarInput(pSrc, nSrcStep, pDst, nDstStep, 4, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiHLSToBGR_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3],
                                                         nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_AP4C4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarToPackedInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiHLSToBGR_8u_AP4C4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pSrc[3], pDst, nDstStep,
                                                         oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_AP4C4R(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_AP4C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiHLSToBGR_8u_AP4R_Ctx(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep,
                                   NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validatePlanarInput(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, 4);
  if (status != NPP_SUCCESS) {
    return status;
  }

  cudaError_t cudaStatus = nppiHLSToBGR_8u_AP4R_kernel(pSrc[0], nSrcStep, pSrc[1], pSrc[2], pSrc[3], pDst[0], pDst[1],
                                                       pDst[2], pDst[3], nDstStep, oSizeROI, nppStreamCtx.hStream);
  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiHLSToBGR_8u_AP4R(const Npp8u *const pSrc[4], int nSrcStep, Npp8u *pDst[4], int nDstStep,
                               NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiHLSToBGR_8u_AP4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
