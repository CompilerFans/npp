// NPP 13.4 P010 / P016 to RGB entry points.
#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiP010ToRgbLaunch(const Npp16u *srcY, int nSrcStep, const Npp16u *srcUV, void *dst, int nDstStep,
                                NppiSize roi, int cs, bool c4, bool wide, Npp16u alpha16, Npp8u alpha8,
                                cudaStream_t stream);
}

namespace {

bool csOk(NppiColorSpace cs) {
  return cs == NPP_BT601JPEG || cs == NPP_BT709HDTV || cs == NPP_BT2020HDTV;
}

NppStatus p010Check(const Npp16u *const pSrc[2], const void *pDst, NppiSize roi, NppiColorSpace cs, int srcStep,
                    int dstStep) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pDst) return NPP_NULL_POINTER_ERROR;
  if (!csOk(cs)) return NPP_NOT_IMPLEMENTED_ERROR;
  if (roi.width <= 0 || roi.height <= 0 || (roi.width & 1) != 0 || (roi.height & 1) != 0) return NPP_SIZE_ERROR;
  if (srcStep <= 0 || dstStep <= 0) return NPP_STEP_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

NppStatus nppiP010ToRGB_8u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  const NppStatus vs = p010Check(pSrc, pDst, oSizeROI, eColorSpace, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiP010ToRgbLaunch(pSrc[0], nSrcStep, pSrc[1], pDst, nDstStep, oSizeROI, (int)eColorSpace, false, false, 0, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP010ToRGB_8u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp8u *pDst, int nDstStep,
                                      NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp8u nAlpha,
                                      NppStreamContext nppStreamCtx) {
  const NppStatus vs = p010Check(pSrc, pDst, oSizeROI, eColorSpace, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiP010ToRgbLaunch(pSrc[0], nSrcStep, pSrc[1], pDst, nDstStep, oSizeROI, (int)eColorSpace, true, false, 0,
                          nAlpha, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP010ToRGB_16u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  const NppStatus vs = p010Check(pSrc, pDst, oSizeROI, eColorSpace, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiP010ToRgbLaunch(pSrc[0], nSrcStep, pSrc[1], pDst, nDstStep, oSizeROI, (int)eColorSpace, false, true, 0, 0,
                          nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP010ToRGB_16u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha,
                                       NppStreamContext nppStreamCtx) {
  const NppStatus vs = p010Check(pSrc, pDst, oSizeROI, eColorSpace, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiP010ToRgbLaunch(pSrc[0], nSrcStep, pSrc[1], pDst, nDstStep, oSizeROI, (int)eColorSpace, true, true, nAlpha,
                          0, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP016ToRGB_16u_P2C3R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppiColorSpace eColorSpace, NppStreamContext nppStreamCtx) {
  return nppiP010ToRGB_16u_P2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eColorSpace, nppStreamCtx);
}

NppStatus nppiP016ToRGB_16u_P2C4R_Ctx(const Npp16u *const pSrc[2], int nSrcStep, Npp16u *pDst, int nDstStep,
                                       NppiSize oSizeROI, NppiColorSpace eColorSpace, Npp16u nAlpha,
                                       NppStreamContext nppStreamCtx) {
  return nppiP010ToRGB_16u_P2C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, eColorSpace, nAlpha, nppStreamCtx);
}

}  // extern "C"
