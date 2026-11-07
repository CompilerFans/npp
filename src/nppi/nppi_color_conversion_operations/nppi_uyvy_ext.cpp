// NPP 13.4 UYVY color-conversion entry points.
//
// Validation replicates the measured reference behaviour: null pointers and
// arrays -> NPP_NULL_POINTER_ERROR; a non-zero oSrcOffset -> NPP_BAD_ARGUMENT_
// ERROR; an unsupported NppiColorSpace -> NPP_NOT_IMPLEMENTED_ERROR; zero or
// odd widths (UYVY is 4:2:2) -> NPP_SIZE_ERROR; non-positive steps ->
// NPP_STEP_ERROR.
#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiUyvy8ToRgbLaunch(const Npp8u *src, int nSrcStep, Npp8u *dst, Npp8u *dst1, Npp8u *dst2, int nDstStep,
                                 NppiSize roi, int mode, int cs, Npp8u alpha, cudaStream_t stream);
cudaError_t nppiRgb8ToUyvyLaunch(const Npp8u *src, const Npp8u *s1, const Npp8u *s2, int nSrcStep, Npp8u *dst,
                                 int nDstStep, NppiSize roi, int mode, int cs, cudaStream_t stream);
cudaError_t nppiUyvy16ToRgb16Launch(const Npp16u *src, int nSrcStep, Npp16u *dst, Npp16u *dst1, Npp16u *dst2,
                                    int nDstStep, NppiSize roi, int mode, int cs, Npp16u alpha,
                                    cudaStream_t stream);
cudaError_t nppiRgb16ToUyvy16Launch(const Npp16u *src, const Npp16u *s1, const Npp16u *s2, int nSrcStep, Npp16u *dst,
                                    int nDstStep, NppiSize roi, int mode, int cs, cudaStream_t stream);
}

namespace {

bool uyvyCsOk(NppiColorSpace cs) {
  return cs == NPP_BT601JPEG || cs == NPP_BT709HDTV || cs == NPP_BT2020HDTV;
}

NppStatus uyvyCheck(NppiPoint off, NppiColorSpace cs, NppiSize roi, int srcStep, int dstStep) {
  if (off.x != 0 || off.y != 0) return NPP_BAD_ARGUMENT_ERROR;
  if (!uyvyCsOk(cs)) return NPP_NOT_IMPLEMENTED_ERROR;
  if (roi.width <= 0 || roi.height <= 0 || (roi.width & 1) != 0) return NPP_SIZE_ERROR;
  if (srcStep <= 0 || dstStep <= 0) return NPP_STEP_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// UYVY 8u -> RGB 8u
// ---------------------------------------------------------------------------
NppStatus nppiUYVY_8u_ToRGB_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy8ToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, 0, (int)eColorSpace, 0,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVY_8u_ToRGB_8u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha,
                                           NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy8ToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, 1, (int)eColorSpace, nAlpha,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVY_8u_ToRGB_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst[3],
                                          const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy8ToRgbLaunch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, 2, (int)eColorSpace, 0,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// UYVY16 16u -> RGB 16u
// ---------------------------------------------------------------------------
NppStatus nppiUYVY16_16u_ToRGB_16u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                              NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy16ToRgb16Launch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, 0, (int)eColorSpace, 0,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVY16_16u_ToRGB_16u_C3AC4R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
                                               int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                               Npp16u nAlpha, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy16ToRgb16Launch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, 1, (int)eColorSpace, nAlpha,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVY16_16u_ToRGB_16u_C3P3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst[3],
                                              const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                              NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvy16ToRgb16Launch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, 2, (int)eColorSpace, 0,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// RGB 8u -> UYVY 8u
// ---------------------------------------------------------------------------
NppStatus nppiRGB_8u_ToUYVY_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                          int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb8ToUyvyLaunch(pSrc, nullptr, nullptr, nSrcStep, pDst, nDstStep, oSizeROI, 0, (int)eColorSpace,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiRGB_8u_ToUYVY_8u_P3C3R_Ctx(const Npp8u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                          Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                          NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb8ToUyvyLaunch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep, oSizeROI, 1, (int)eColorSpace,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiRGBA_8u_ToUYVY_8u_AC4C3R_Ctx(const Npp8u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb8ToUyvyLaunch(pSrc, nullptr, nullptr, nSrcStep, pDst, nDstStep, oSizeROI, 2, (int)eColorSpace,
                           nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// RGB 16u -> UYVY16 16u
// ---------------------------------------------------------------------------
NppStatus nppiRGB_16u_ToUYVY16_16u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                              NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb16ToUyvy16Launch(pSrc, nullptr, nullptr, nSrcStep, pDst, nDstStep, oSizeROI, 0, (int)eColorSpace,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiRGB_16u_ToUYVY16_16u_P3C3R_Ctx(const Npp16u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                              Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                              NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb16ToUyvy16Launch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nDstStep, oSizeROI, 1, (int)eColorSpace,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiRGBA_16u_ToUYVY16_16u_AC4C3R_Ctx(const Npp16u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                                Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                                NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvyCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiRgb16ToUyvy16Launch(pSrc, nullptr, nullptr, nSrcStep, pDst, nDstStep, oSizeROI, 2, (int)eColorSpace,
                              nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // extern "C"
