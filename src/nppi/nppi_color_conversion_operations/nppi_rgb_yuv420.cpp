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
  if (rDstStep[0] <= 0 || rDstStep[1] <= 0 || rDstStep[2] <= 0) {
    return NPP_STEP_ERROR;
  }
  if (rDstStep[0] < oSizeROI.width || rDstStep[1] < oSizeROI.width / 2 ||
      rDstStep[2] < oSizeROI.width / 2) {
    return NPP_STEP_ERROR;
  }
  return NPP_NO_ERROR;
}

// Measured reference extent rule: the converted extent is (w&~1)x(h&~1)
// and an odd width or height returns NPP_DOUBLE_SIZE_WARNING.
static inline NppiSize yuv420FwdRoi(NppiSize roi) { roi.width &= ~1; roi.height &= ~1; return roi; }
static inline NppStatus yuv420FwdStatus(cudaError_t s, NppiSize roi) {
  if (s != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return ((roi.width | roi.height) & 1) ? NPP_DOUBLE_SIZE_WARNING : NPP_NO_ERROR;
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
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv420FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiRGBToYUV420_8u_C3P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                      rDstStep[2], roi, nppStreamCtx.hStream);

  return yuv420FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiRGBToYUV420_8u_C3P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                   NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
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
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv420FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiRGBToYUV420_8u_P3R_kernel(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1],
                                    pDst[2], rDstStep[2], roi, nppStreamCtx.hStream);

  return yuv420FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiRGBToYUV420_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                 NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
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
  // An empty ROI is a no-op.
  if (oSizeROI.width == 0 || oSizeROI.height == 0) {
    return NPP_NO_ERROR;
  }

  const NppiSize roi = yuv420FwdRoi(oSizeROI);
  cudaError_t cudaStatus =
      nppiBGRToYUV420_8u_AC4P3R_kernel(pSrc, nSrcStep, pDst[0], rDstStep[0], pDst[1], rDstStep[1], pDst[2],
                                       rDstStep[2], roi, nppStreamCtx.hStream);

  return yuv420FwdStatus(cudaStatus, oSizeROI);
}

NppStatus nppiBGRToYUV420_8u_AC4P3R(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[3], int rDstStep[3],
                                    NppiSize oSizeROI) {
  NppStreamContext ctx{};
  ctx.hStream = nppGetStream();
  return nppiBGRToYUV420_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, ctx);
}
