#include "npp.h"
#include <cstring>
#include <cuda_runtime.h>

// Kernel declarations

extern "C" {
cudaError_t nppiRGBToYUV_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_AC4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                        cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_P3R_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                       Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                         int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiRGBToYUV_8u_AC4P4R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, Npp8u *pDstU, Npp8u *pDstV,
                                          Npp8u *pDstA, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
cudaError_t nppiYUVToRGB_8u_C3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                       cudaStream_t stream);
}

NppStatus nppiRGBToYUV_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 3 || nDstStep < oSizeROI.width * 3) {
    return NPP_STEP_ERROR;
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
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 4 || nDstStep < oSizeROI.width * 4) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus = nppiRGBToYUV_8u_AC4R_kernel(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_AC4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                                 NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus =
      nppiRGBToYUV_8u_P3R_kernel(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI,
                                 nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int nDstStep,
                              NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                   NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 3 || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
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

NppStatus nppiRGBToYUV_8u_AC4P4R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI,
                                    NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2] || !pDst[3]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 4 || nDstStep < oSizeROI.width) {
    return NPP_STEP_ERROR;
  }

  cudaError_t cudaStatus = nppiRGBToYUV_8u_AC4P4R_kernel(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], pDst[3], nDstStep,
                                                        oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_SUCCESS : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV_8u_AC4P4R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[4], int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV_8u_AC4P4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiYUVToRGB_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                  NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep < oSizeROI.width * 3 || nDstStep < oSizeROI.width * 3) {
    return NPP_STEP_ERROR;
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
