#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYUV420_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                            Npp8u *pDstU, int nDstUStep, Npp8u *pDstV, int nDstVStep,
                                            NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiRGBToYUV420_8u_P3R_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                          Npp8u *pDstY, int nDstYStep, Npp8u *pDstU, int nDstUStep, Npp8u *pDstV,
                                          int nDstVStep, NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiBGRToYUV420_8u_AC4P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                             Npp8u *pDstU, int nDstUStep, Npp8u *pDstV, int nDstVStep,
                                             NppiSize oSizeROI, cudaStream_t stream);
}

static NppStatus validateYUV420Inputs(const Npp8u *const pDst[3], const int rDstStep[3], NppiSize oSizeROI) {
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

NppStatus nppiRGBToYUV420_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYUV420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYUV420_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                      rDstStep[2], oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV420_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV420_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYUV420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiRGBToYUV420_8u_P3R_kernel(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1],
                                    pDst[2], rDstStep[2], oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiRGBToYUV420_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiRGBToYUV420_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiBGRToYUV420_8u_AC4P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                        NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (nSrcStep <= 0) {
    return NPP_STEP_ERROR;
  }

  NppStatus status = validateYUV420Inputs(pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }

  cudaError_t cudaStatus =
      nppiBGRToYUV420_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                       rDstStep[2], oSizeROI, nppStreamCtx.hStream);

  return (cudaStatus == cudaSuccess) ? NPP_NO_ERROR : NPP_CUDA_KERNEL_EXECUTION_ERROR;
}

NppStatus nppiBGRToYUV420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiBGRToYUV420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}
