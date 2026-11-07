#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiRGBToYUV422_8u_C3P3R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDstY, int nDstYStep,
                                            Npp8u *pDstU, int nDstUStep, Npp8u *pDstV, int nDstVStep,
                                            NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiRGBToYUV422_8u_P3R_kernel(const Npp8u *pSrcR, const Npp8u *pSrcG, const Npp8u *pSrcB, int nSrcStep,
                                          Npp8u *pDstY, int nDstYStep, Npp8u *pDstU, int nDstUStep, Npp8u *pDstV,
                                          int nDstVStep, NppiSize oSizeROI, cudaStream_t stream);

cudaError_t nppiRGBToYUV422_8u_C3C2R_kernel(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, cudaStream_t stream);
}

static NppStatus validateYUV422Planar(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                      NppiSize oSizeROI) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  if (!rDstStep) {
    return NPP_NULL_POINTER_ERROR;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep <= 0 || rDstStep[0] <= 0 || rDstStep[1] <= 0 || rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  if (rDstStep[0] < oSizeROI.width || rDstStep[1] < oSizeROI.width / 2 ||
      rDstStep[2] < oSizeROI.width / 2) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

static NppStatus validateYUV422PlanarSrc(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                         NppiSize oSizeROI) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2]) {
    return NPP_NULL_POINTER_ERROR;
  }
  return validateYUV422Planar(pSrc[0], nSrcStep, pDst, rDstStep, oSizeROI);
}

// Measured reference extent rule: the luma extent is w&~1 columns on every
// row and an odd width returns NPP_DOUBLE_SIZE_WARNING.
static inline NppiSize yuv422FwdRoi(NppiSize roi) { roi.width &= ~1; return roi; }
static inline NppStatus yuv422FwdStatus(cudaError_t s, NppiSize roi) {
  if (s != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return (roi.width & 1) ? NPP_DOUBLE_SIZE_WARNING: NPP_NO_ERROR;
}

NppStatus nppiRGBToYUV422_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYUV422Planar(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv422FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiRGBToYUV422_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                      rDstStep[2], roi, nppStreamCtx.hStream);

  return yuv422FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiRGBToYUV422_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiRGBToYUV422_8u_C3P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV422_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                     NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  NppStatus status = validateYUV422PlanarSrc(pSrc, nSrcStep, pDst, rDstStep, oSizeROI);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv422FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiRGBToYUV422_8u_P3R_kernel(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1],
                                    pDst[2], rDstStep[2], roi, nppStreamCtx.hStream);

  return yuv422FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiRGBToYUV422_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                 NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiRGBToYUV422_8u_P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}

NppStatus nppiRGBToYUV422_8u_C3C2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) {
    return NPP_NULL_POINTER_ERROR;
  }
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }
  if (oSizeROI.width < 0 || oSizeROI.height < 0) {
    return NPP_SIZE_ERROR;
  }
  if (nSrcStep <= 0 || nDstStep <= 0) {
    return NPP_STEP_ERROR;
  }
  if (nDstStep < oSizeROI.width * 2) {
    return NPP_STEP_ERROR;
  }

  const NppiSize roi = yuv422FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiRGBToYUV422_8u_C3C2R_kernel(pSrc, nSrcStep, pDst, nDstStep, roi, nppStreamCtx.hStream);

  return yuv422FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiRGBToYUV422_8u_C3C2R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiRGBToYUV422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, ctx);
}
