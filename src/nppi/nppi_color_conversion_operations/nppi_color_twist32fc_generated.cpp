// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwist32fC


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwist32fC_16f_C4IR_Ctx(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCIR<Npp16f>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32fC_16f_C4R_Ctx(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCR<Npp16f>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32fC_8u_C4IR_Ctx(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCIR<Npp8u>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32fC_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCR<Npp8u>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist32fC_16f_C4IR(Npp16f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32fC_16f_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}

NppStatus nppiColorTwist32fC_16f_C4R(const Npp16f * pSrc, int nSrcStep, Npp16f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32fC_16f_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}

NppStatus nppiColorTwist32fC_8u_C4IR(Npp8u * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32fC_8u_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}

NppStatus nppiColorTwist32fC_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist32fC_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}
