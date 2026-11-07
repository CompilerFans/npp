// NPP 13.4 UYVP_10u color-conversion entry points.
#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiUyvpToRgbLaunch(const Npp8u *src, int nSrcStep, void *dst, void *dst1, void *dst2, int nDstStep,
                                NppiSize roi, int cs, int mode, int outKind, float alpha, cudaStream_t stream);
cudaError_t nppiRgbToUyvpLaunch(const void *src, const void *s1, const void *s2, int nSrcStep, Npp8u *dst,
                                Npp8u *p1, Npp8u *p2, int nDstStep, NppiSize roi, int cs, int mode, int inKind,
                                bool planar, cudaStream_t stream);
}

namespace {

bool csOk(NppiColorSpace cs) {
  return cs == NPP_BT601JPEG || cs == NPP_BT709HDTV || cs == NPP_BT2020HDTV;
}

NppStatus uyvpCheck(NppiPoint off, NppiColorSpace cs, NppiSize roi, int srcStep, int dstStep) {
  if (off.x != 0 || off.y != 0) return NPP_BAD_ARGUMENT_ERROR;
  if (!csOk(cs)) return NPP_NOT_IMPLEMENTED_ERROR;
  if (roi.width <= 0 || roi.height <= 0 || (roi.width & 1) != 0) return NPP_SIZE_ERROR;
  if (srcStep <= 0 || dstStep <= 0) return NPP_STEP_ERROR;
  return NPP_NO_ERROR;
}

NppStatus encDispatch(const void *src, const void *s1, const void *s2, int nSrcStep, Npp8u *dst, Npp8u *p1, Npp8u *p2,
                      int nDstStep, NppiSize roi, NppiColorSpace cs, int mode, int inKind, bool planar,
                      NppStreamContext ctx) {
  if (nppiRgbToUyvpLaunch(src, s1, s2, nSrcStep, (Npp8u *)dst, (Npp8u *)p1, (Npp8u *)p2, nDstStep, roi,
                          (int)cs, mode, inKind, planar, ctx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// RGB(A) 8u -> UYVP_10u
// ---------------------------------------------------------------------------
NppStatus nppiRGB_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 0, 0,
                     false, nppStreamCtx);
}

NppStatus nppiRGB_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                           Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 1,
                     0, false, nppStreamCtx);
}

NppStatus nppiRGB_8u_ToUYVP_10u_C3P3R_Ctx(const Npp8u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                           Npp8u *pDst[3], int nDstStep, NppiSize oSizeROI,
                                           NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, eColorSpace, 0,
                     0, true, nppStreamCtx);
}

NppStatus nppiRGBA_8u_ToUYVP_10u_AC4C3R_Ctx(const Npp8u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                             NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 2, 0,
                     false, nppStreamCtx);
}

// ---------------------------------------------------------------------------
// RGB(A) 16u -> UYVP_10u
// ---------------------------------------------------------------------------
NppStatus nppiRGB_16u_ToUYVP_10u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 0, 1,
                     false, nppStreamCtx);
}

NppStatus nppiRGB_16u_ToUYVP_10u_P3C3R_Ctx(const Npp16u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 1,
                     1, false, nppStreamCtx);
}

NppStatus nppiRGBA_16u_ToUYVP_10u_AC4C3R_Ctx(const Npp16u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                              Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                              NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 2, 1,
                     false, nppStreamCtx);
}

// ---------------------------------------------------------------------------
// RGB(A) 16f -> UYVP_10u
// ---------------------------------------------------------------------------
NppStatus nppiRGB_16f_ToUYVP_10u_C3C3R_Ctx(const Npp16f *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 0, 2,
                     false, nppStreamCtx);
}

NppStatus nppiRGB_16f_ToUYVP_10u_P3C3R_Ctx(const Npp16f *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 1,
                     2, false, nppStreamCtx);
}

NppStatus nppiRGBA_16f_ToUYVP_10u_AC4C3R_Ctx(const Npp16f *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                              NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  return encDispatch(pSrc, nullptr, nullptr, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, eColorSpace, 2, 2,
                     false, nppStreamCtx);
}

// ---------------------------------------------------------------------------
// UYVP_10u -> RGB(A) 8u
// ---------------------------------------------------------------------------
NppStatus nppiUYVP_10u_ToRGB_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                           int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 0, 0, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGB_8u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u alpha,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 1, 0,
                          (float)alpha, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGB_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst[3],
                                           const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                           NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, (int)eColorSpace, 2, 0, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// UYVP_10u -> RGB(A) 16u
// ---------------------------------------------------------------------------
NppStatus nppiUYVP_10u_ToRGB_16u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 0, 1, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGB_16u_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst,
                                             int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                             Npp16u alpha, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 1, 1,
                          (float)alpha, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGB_16u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16u *pDst[3],
                                            const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, (int)eColorSpace, 2, 1, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// UYVP_10u -> RGB(A) 16f
// ---------------------------------------------------------------------------
NppStatus nppiUYVP_10u_ToRGB_16f_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 0, 2, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGB_16f_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f *const pDst[3],
                                            const int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, oSizeROI, (int)eColorSpace, 2, 2, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToRGBA_16f_C3AC4R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp16f *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppiColorSpace eColorSpace,
                                              Npp32f nAlphaValue, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = uyvpCheck(oSrcOffset, eColorSpace, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpToRgbLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, oSizeROI, (int)eColorSpace, 1, 2,
                          nAlphaValue, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // extern "C"
