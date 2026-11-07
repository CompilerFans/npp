// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwist32f


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwist32f_16f_C1IR_Ctx(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16f, 1, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C1R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16f, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C2IR_Ctx(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16f, 2, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C2R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16f, 2, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C3IR_Ctx(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16f, 3, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C3R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16f, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C4IR_Ctx(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp16f, 4, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp16f, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_AC4IR_Ctx(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8s, 4, 1>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_AC4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8s, 4, 1>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C1IR_Ctx(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8s, 1, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C1R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8s, 1, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C2IR_Ctx(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8s, 2, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C2R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8s, 2, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C3IR_Ctx(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8s, 3, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C3R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8s, 3, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C4IR_Ctx(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIR<Npp8s, 4, 0>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_C4R_Ctx(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistR<Npp8s, 4, 0>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_IP3R_Ctx(Npp8s * const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistIP3<Npp8s>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_8s_P3R_Ctx(const Npp8s * const pSrc[3], int nSrcStep, Npp8s * const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistP3<Npp8s>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32f_16f_C1IR(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C1R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C2IR(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C2R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C3IR(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C3R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C4IR(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_16f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_AC4IR(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_AC4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_AC4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_AC4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C1IR(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C1IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C1R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C2IR(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C2IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C2R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C3IR(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C3IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C3R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C4IR(Npp8s * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_C4R(const Npp8s * pSrc, int nSrcStep, Npp8s * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_IP3R(Npp8s * const pSrcDst[3], int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_IP3R_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, nppStreamCtx);
}

NppStatus nppiColorTwist32f_8s_P3R(const Npp8s * const pSrc[3], int nSrcStep, Npp8s * const pDst[3], int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32f_8s_P3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx);
}
