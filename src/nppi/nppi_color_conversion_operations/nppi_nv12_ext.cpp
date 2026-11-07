// NPP 13.4 NV12 extended color-conversion entry points.
//
// Validation behaviour replicates the measured NVIDIA NPP 13.4 host checks:
//   * dedicated 601 C3 conversions: null -> -8, zero ROI -> success, odd width
//     and a source step that is not a multiple of 4 -> NPP_SIZE_ERROR, a too
//     small destination step -> NPP_STEP_ERROR;
//   * BT2020 / C4 variants: null -> -8, zero or odd-sized ROI -> NPP_SIZE_ERROR,
//     an unsupported NppiColorSpace -> NPP_NOT_IMPLEMENTED_ERROR, no step checks;
//   * RGB -> NV12 and YUV420 -> NV12: null -> -8, zero/odd-sized ROI -> -6,
//     unsupported color space -> -3, no step checks.
#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiNv12ExtToRgbLaunch(const Npp8u *srcY, int nSrcStep, const Npp8u *srcUV, Npp8u *dst, int nDstStep,
                                   NppiSize roi, int coefSet, bool bgr, int dstC, Npp8u alpha, cudaStream_t stream);
cudaError_t nppiRgbToNv12Launch(const Npp8u *src0, const Npp8u *src1, const Npp8u *src2, int nSrcStep, bool planar,
                                Npp8u *dstY, Npp8u *dstUV, int nDstStep, NppiSize roi, int colorSpace,
                                cudaStream_t stream);
cudaError_t nppiYuv420ToNv12Launch(const Npp8u *const src[3], const int srcStep[3], Npp8u *dstY, Npp8u *dstUV,
                                   int nDstStep, NppiSize roi, cudaStream_t stream);
}

namespace {

bool colorSpaceSupported(NppiColorSpace cs) {
  return cs == NPP_BT601JPEG || cs == NPP_BT709HDTV || cs == NPP_BT2020HDTV;
}

int coefSetForColorSpace(NppiColorSpace cs) {
  switch (cs) {
    case NPP_BT601JPEG: return 3;
    case NPP_BT709HDTV: return 1;
    default: return 2;
  }
}

NppStatus nv12ToRgbCommon(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep, NppiSize roi,
                          NppStreamContext ctx, int coefSet, bool bgr, int dstC, Npp8u alpha, bool strictSize,
                          bool stepChecks) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) return NPP_NULL_POINTER_ERROR;
  if (!strictSize) {
    if (roi.width == 0 || roi.height == 0) return NPP_NO_ERROR;
    if (roi.width < 0 || roi.height < 0) return NPP_SIZE_ERROR;
    if (roi.width % 2 != 0) return NPP_SIZE_ERROR;
    if (stepChecks) {
      if (nSrcStep % 4 != 0) return NPP_SIZE_ERROR;
      if (nDstStep < roi.width * dstC) return NPP_STEP_ERROR;
    }
  } else {
    if (roi.width <= 0 || roi.height <= 0 || roi.width % 2 != 0 || roi.height % 2 != 0) return NPP_SIZE_ERROR;
  }
  cudaError_t st = nppiNv12ExtToRgbLaunch(pSrc[0], nSrcStep, pSrc[1], pDst, nDstStep, roi, coefSet, bgr, dstC, alpha,
                                          ctx.hStream);
  if (st != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

NppStatus nppiNV12ToRGB_601_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nv12ToRgbCommon(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, 0, false, 3, 0xFF, false, true);
}

NppStatus nppiNV12ToBGR_601_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int rSrcStep, Npp8u *pDst, int nDstStep,
                                          NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nv12ToRgbCommon(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, 0, true, 3, 0xFF, false, true);
}

NppStatus nppiNV12ToRGB_BT2020_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nv12ToRgbCommon(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, 2, false, 3, 0xFF, true, false);
}

NppStatus nppiNV12ToBGR_BT2020_8u_P2C3R_Ctx(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                            NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nv12ToRgbCommon(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, 2, true, 3, 0xFF, true, false);
}

NppStatus nppiNV12ToRGB_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha,
                                     NppStreamContext nppStreamCtx) {
  if (!colorSpaceSupported(eColorSpace)) return NPP_NOT_IMPLEMENTED_ERROR;
  return nv12ToRgbCommon(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, coefSetForColorSpace(eColorSpace),
                         false, 4, nAlpha, true, false);
}

NppStatus nppiNV12ToBGR_8u_P2C4R_Ctx(const Npp8u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                     NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha,
                                     NppStreamContext nppStreamCtx) {
  if (!colorSpaceSupported(eColorSpace)) return NPP_NOT_IMPLEMENTED_ERROR;
  return nv12ToRgbCommon(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx, coefSetForColorSpace(eColorSpace),
                         true, 4, nAlpha, true, false);
}

NppStatus nppiRGBToNV12_8u_C3P2R_Ctx(const Npp8u *pSrc, int nSrcStep, Npp8u *pDst[2], int nDstStep, NppiSize oSizeROI,
                                     NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1]) return NPP_NULL_POINTER_ERROR;
  if (!colorSpaceSupported(eColorSpace)) return NPP_NOT_IMPLEMENTED_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0)
    return NPP_SIZE_ERROR;
  cudaError_t st = nppiRgbToNv12Launch(pSrc, nullptr, nullptr, nSrcStep, false, pDst[0], pDst[1], nDstStep, oSizeROI,
                                       (int)eColorSpace, nppStreamCtx.hStream);
  if (st != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiRGBToNV12_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], int nSrcStep, Npp8u *pDst[2], int nDstStep,
                                     NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1]) return NPP_NULL_POINTER_ERROR;
  if (!colorSpaceSupported(eColorSpace)) return NPP_NOT_IMPLEMENTED_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0)
    return NPP_SIZE_ERROR;
  cudaError_t st = nppiRgbToNv12Launch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, true, pDst[0], pDst[1], nDstStep,
                                       oSizeROI, (int)eColorSpace, nppStreamCtx.hStream);
  if (st != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiYUV420ToNV12_8u_P3P2R_Ctx(const Npp8u *const pSrc[3], const int aSrcStep[3], Npp8u *pDst[2],
                                        int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst || !pDst[0] || !pDst[1] || !aSrcStep)
    return NPP_NULL_POINTER_ERROR;
  if (oSizeROI.width <= 0 || oSizeROI.height <= 0 || oSizeROI.width % 2 != 0 || oSizeROI.height % 2 != 0)
    return NPP_SIZE_ERROR;
  cudaError_t st = nppiYuv420ToNv12Launch(pSrc, aSrcStep, pDst[0], pDst[1], nDstStep, oSizeROI,
                                          nppStreamCtx.hStream);
  if (st != cudaSuccess) return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // extern "C"
