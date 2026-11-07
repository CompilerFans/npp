// NPP 13.4 UYVP_10u transcoding entry points.
//
// These APIs are declared by the 13.4 headers but are not exported by the
// reference NPP 13.2.0.58 library.  They are implemented from the documented
// format layouts (see nppi_uyvp_transcode.cu).
#include "npp.h"
#include <cuda_runtime.h>

extern "C" {
cudaError_t nppiUyvpTranscodeToLaunch(const void *s0, const void *s1, const void *s2, int st0, int st1, int kind,
                                      Npp8u *dst, Npp8u *d1, Npp8u *d2, int dstStep, NppiSize roi, int wide,
                                      cudaStream_t stream);
cudaError_t nppiUyvpTranscodeFromLaunch(const Npp8u *src, int srcStep, Npp8u *d0, Npp8u *d1, Npp8u *d2, int dstStep,
                                        int dstStep1, NppiSize roi, int kind, cudaStream_t stream);
}

namespace {

NppStatus tcCheck(NppiPoint off, NppiSize roi, int srcStep, int dstStep) {
  if (off.x != 0 || off.y != 0) return NPP_BAD_ARGUMENT_ERROR;
  if (roi.width <= 0 || roi.height <= 0 || (roi.width & 1) != 0) return NPP_SIZE_ERROR;
  if (srcStep <= 0 || dstStep <= 0) return NPP_STEP_ERROR;
  return NPP_NO_ERROR;
}

}  // namespace

extern "C" {

NppStatus nppiNV12_8u_ToUYVP_10u_P2C3R_Ctx(const Npp8u *const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !nSrcStep || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep[0], nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc[0], pSrc[1], nullptr, nSrcStep[0], nSrcStep[1], 0, pDst, nullptr, nullptr,
                                nDstStep, oSizeROI, 0, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiNV16_8u_ToUYVP_10u_P2C3R_Ctx(const Npp8u *const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !nSrcStep || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep[0], nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc[0], pSrc[1], nullptr, nSrcStep[0], nSrcStep[1], 1, pDst, nullptr, nullptr,
                                nDstStep, oSizeROI, 0, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP010_16u_ToUYVP_10u_P3C3R_Ctx(const Npp16u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, nSrcStep, 4, pDst, nullptr, nullptr, nDstStep,
                                oSizeROI, 1, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP208_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                            int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc, nullptr, nullptr, nSrcStep, 0, 5, pDst, nullptr, nullptr, nDstStep, oSizeROI, 0,
                                nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiP210_16u_ToUYVP_10u_P2C3R_Ctx(const Npp16u *const pSrc[2], const int nSrcStep[2], NppiPoint oSrcOffset,
                                             Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !nSrcStep || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep[0], nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc[0], pSrc[1], nullptr, nSrcStep[0], nSrcStep[1], 3, pDst, nullptr, nullptr,
                                nDstStep, oSizeROI, 1, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiI444_8u_ToUYVP_10u_P3C3R_Ctx(const Npp8u *const pSrc[3], const int nSrcStep, NppiPoint oSrcOffset,
                                            Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pSrc[0] || !pSrc[1] || !pSrc[2] || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc[0], pSrc[1], pSrc[2], nSrcStep, nSrcStep, 6, pDst, nullptr, nullptr, nDstStep,
                                oSizeROI, 0, nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiV210_10u_ToUYVP_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc, nullptr, nullptr, nSrcStep, 0, 8, pDst, nullptr, nullptr, nDstStep, oSizeROI, 0,
                                nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiY410_10u_ToUYVP_10u_C3C3R_Ctx(const Npp16u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc, nullptr, nullptr, nSrcStep, 0, 9, pDst, nullptr, nullptr, nDstStep, oSizeROI, 1,
                                nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiYUV444_8u_ToUYVP_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeToLaunch(pSrc, nullptr, nullptr, nSrcStep, 0, 7, pDst, nullptr, nullptr, nDstStep, oSizeROI, 0,
                                nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

// ---------------------------------------------------------------------------
// From UYVP.
// ---------------------------------------------------------------------------
NppStatus nppiUYVP_10u_ToI444_8u_C3P3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset,
                                             Npp8u *const pDst[3], const int nDstStep, NppiSize oSizeROI,
                                             NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !pDst[2]) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, nSrcStep, pDst[0], pDst[1], pDst[2], nDstStep, nDstStep, oSizeROI, 0,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToNV16_8u_C3P2R_Ctx(const Npp8u *const pSrc, const int nSrcStep, NppiPoint oSrcOffset,
                                            Npp8u *pDst[2], const int nDstStep[2], NppiSize oSizeROI,
                                            NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst || !pDst[0] || !pDst[1] || !nDstStep) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep[0]);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, nSrcStep, pDst[0], pDst[1], nullptr, nDstStep[0], nDstStep[1], oSizeROI, 1,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToP208_8u_C3C3R_Ctx(const Npp8u *pSrc, int srcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                            int dstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, srcStep, dstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, srcStep, pDst, nullptr, nullptr, dstStep, 0, oSizeROI, 2,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToV210_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, 0, oSizeROI, 3,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToY410_10u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                             int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, 0, oSizeROI, 4,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

NppStatus nppiUYVP_10u_ToYUV444_8u_C3C3R_Ctx(const Npp8u *pSrc, int nSrcStep, NppiPoint oSrcOffset, Npp8u *pDst,
                                              int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  if (!pSrc || !pDst) return NPP_NULL_POINTER_ERROR;
  const NppStatus vs = tcCheck(oSrcOffset, oSizeROI, nSrcStep, nDstStep);
  if (vs != NPP_NO_ERROR) return vs;
  if (nppiUyvpTranscodeFromLaunch(pSrc, nSrcStep, pDst, nullptr, nullptr, nDstStep, 0, oSizeROI, 5,
                                  nppStreamCtx.hStream) != cudaSuccess)
    return NPP_CUDA_KERNEL_EXECUTION_ERROR;
  return NPP_NO_ERROR;
}

}  // extern "C"
