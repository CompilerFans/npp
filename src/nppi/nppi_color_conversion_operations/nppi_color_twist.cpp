// nppiColorTwist32f entry points for the 8u/16u/16s image types.
//
// These were originally hand-written kernels with divergent semantics; they
// now delegate to the shared colortwist executor, whose behaviour is pinned
// by the reference outputs of NVIDIA NPP 13.4:
//   * missing input channels read as 0 (no replication)
//   * C4 alpha copied, AC4 alpha untouched
//   * integer outputs: truncate toward zero, then saturate to the dtype range;
//     packed CH>=3 adds a +0.5f rounding bias first (round half up)
//   * validation: negative ROI -> SIZE_ERROR; dst step unsigned-range check;
//     src step unchecked; planar signed both-step check
#include "npp.h"
#include "nppi_color_twist_executor.h"
#include <cuda_runtime.h>

extern "C" {

NppStatus nppiColorTwist32f_8u_C1R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8u, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C1R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8u, 2, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C2R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8u, 3, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C3R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8u, 4, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C4R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4R_Ctx(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8u, 4, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                             aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_AC4R(const Npp8u *pSrc, int nSrcStep,
                                      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C1IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8u, 1, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C1IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C2IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8u, 2, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C2IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C3IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8u, 3, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C3IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_C4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8u, 4, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_C4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_AC4IR_Ctx(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8u, 4, true>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                               nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_AC4IR(Npp8u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_P3R_Ctx(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistP3<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist,
                                            nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_P3R(const Npp8u *const pSrc[3], int nSrcStep,
                                      Npp8u *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8u_IP3R_Ctx(Npp8u *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIP3<Npp8u>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                             nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8u_IP3R(Npp8u *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8u_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16u, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C1R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16u, 2, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C2R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16u, 3, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C3R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4R_Ctx(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16u, 4, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                             aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_AC4R(const Npp16u *pSrc, int nSrcStep,
                                      Npp16u *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C1IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16u, 1, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C1IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C2IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16u, 2, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C2IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_C3IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16u, 3, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_C3IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_AC4IR_Ctx(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16u, 4, true>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                               nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_AC4IR(Npp16u *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_P3R_Ctx(const Npp16u *const pSrc[3], int nSrcStep,
                                      Npp16u *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistP3<Npp16u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist,
                                            nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_P3R(const Npp16u *const pSrc[3], int nSrcStep,
                                      Npp16u *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16u_IP3R_Ctx(Npp16u *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIP3<Npp16u>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                             nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16u_IP3R(Npp16u *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16u_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16s, 1, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C1R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16s, 2, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C2R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16s, 3, false>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                              aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C3R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4R_Ctx(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16s, 4, true>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI,
                                                             aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_AC4R(const Npp16s *pSrc, int nSrcStep,
                                      Npp16s *pDst, int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C1IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16s, 1, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C1IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C2IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16s, 2, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C2IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_C3IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16s, 3, false>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                                nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_C3IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_AC4IR_Ctx(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16s, 4, true>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                                               nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_AC4IR(Npp16s *pSrcDst, int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_P3R_Ctx(const Npp16s *const pSrc[3], int nSrcStep,
                                      Npp16s *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistP3<Npp16s>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist,
                                            nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_P3R(const Npp16s *const pSrc[3], int nSrcStep,
                                      Npp16s *const pDst[3], int nDstStep, NppiSize oSizeROI,
                                      const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16s_IP3R_Ctx(Npp16s *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIP3<Npp16s>(pSrcDst, nSrcDstStep, oSizeROI, aTwist,
                                             nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16s_IP3R(Npp16s *const pSrcDst[3], int nSrcDstStep,
                                      NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  const NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16s_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}
}
