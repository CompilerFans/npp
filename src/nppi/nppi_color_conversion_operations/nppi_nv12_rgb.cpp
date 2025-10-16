#include "npp.h"
#include <cuda_runtime.h>

// Kernel declarations

extern "C" {
cudaError_t nppiNV12ToRGB_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV, int nSrcUVStep,
                                          Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV, int nSrcUVStep,
                                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV,
                                                        int nSrcUVStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                                        const Npp32f aTwist[3][4], cudaStream_t stream);

cudaError_t nppiNV12ToBGR_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV, int nSrcUVStep,
                                          Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiNV12ToBGR_709CSC_8u_P2C3R_kernel(const Npp8u *pSrcY, int nSrcYStep, const Npp8u *pSrcUV, int nSrcUVStep,
                                                 Npp8u *pDst, int nDstStep, NppiSize oSizeROI, cudaStream_t stream);
}

NppStatus nppiNV12ToRGB_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  if (rSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // NV12 format requires even dimensions for proper chroma handling
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // Call GPU kernel
  cudaError_t cudaStatus = nppiNV12ToRGB_8u_P2C3R_kernel(pSrc[0], rSrcStep, // Y plane
                                                         pSrc[1], rSrcStep, // UV plane (interleaved)
                                                         pDst, nDstStep,    // RGB output
                                                         oSizeROI, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiNV12ToRGB_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiNV12ToRGB_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  if (rSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // NV12 format requires even dimensions
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // CallBT.709 GPUkernel
  cudaError_t cudaStatus = nppiNV12ToRGB_709CSC_8u_P2C3R_kernel(pSrc[0], rSrcStep, // Y plane
                                                                pSrc[1], rSrcStep, // UV plane
                                                                pDst, nDstStep,    // RGB output
                                                                oSizeROI, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiNV12ToRGB_709CSC_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiNV12ToRGB_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNV12ToRGB_709HDTV_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiNV12ToRGB_709HDTV_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst,
                                                   int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4],
                                                   NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!aTwist) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (!aSrcStep || aSrcStep[0] <= 0 || aSrcStep[1] <= 0) {
    return NPP_STEP_ERROR;
  }

  if (nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // NV12 format requires even dimensions for proper chroma handling
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // CallColorTwist GPUkernel
  cudaError_t cudaStatus = nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_kernel(pSrc[0], aSrcStep[0], // Y plane
                                                                       pSrc[1], aSrcStep[1], // UV plane (interleaved)
                                                                       pDst, nDstStep,       // RGB output
                                                                       oSizeROI,
                                                                       aTwist, // Color transformation matrix
                                                                       nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_SUCCESS;
}

NppStatus nppiNV12ToRGB_8u_ColorTwist32f_P2C3R(const Npp8u *const pSrc[2], int aSrcStep[2], Npp8u *pDst, int nDstStep,
                                               NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext ctx;
  ctx.hStream = 0; // Default stream
  return nppiNV12ToRGB_8u_ColorTwist32f_P2C3R_Ctx(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, aTwist, ctx);
}

//=============================================================================
// NV12 to BGR conversion functions
//=============================================================================
NppStatus nppiNV12ToBGR_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  if (rSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // NV12 format requires even dimensions
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // Call BGR kernel
  cudaError_t cudaStatus = nppiNV12ToBGR_8u_P2C3R_kernel(pSrc[0], rSrcStep, // Y plane
                                                         pSrc[1], rSrcStep, // UV plane (interleaved)
                                                         pDst, nDstStep,    // BGR output
                                                         oSizeROI, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiNV12ToBGR_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                 NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiNV12ToBGR_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  // Parameter validation
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }

  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  if (rSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }

  // NV12 format requires even dimensions
  if (oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0) {
    return NPP_WRONG_INTERSECTION_ROI_ERROR;
  }

  // Call BT.709 BGR kernel
  cudaError_t cudaStatus = nppiNV12ToBGR_709CSC_8u_P2C3R_kernel(pSrc[0], rSrcStep, // Y plane
                                                                pSrc[1], rSrcStep, // UV plane
                                                                pDst, nDstStep,    // BGR output
                                                                oSizeROI, nppStreamCtx.hStream);

  if (cudaStatus != cudaSuccess) {
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  }

  return NPP_NO_ERROR;
}

NppStatus nppiNV12ToBGR_709CSC_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                        NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}

NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                             NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppiNV12ToBGR_709CSC_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiNV12ToBGR_709HDTV_8u_P2C3R(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                         NppiSize oSizeROI) {
  NppStreamContext ctx;
  ctx.hStream = 0;
  return nppiNV12ToBGR_709HDTV_8u_P2C3R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
