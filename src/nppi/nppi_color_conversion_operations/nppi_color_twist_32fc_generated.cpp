// Generated API wrappers -- do not edit by hand.
// family: nppiColorTwist_32fC


#include "npp.h"
#include "nppi_color_twist_executor.h"

NppStatus nppiColorTwist_32fC_C4IR_Ctx(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCIR<Npp32f>(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist_32fC_C4R_Ctx(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4], NppStreamContext nppStreamCtx) {
  return nppi::colortwist::twistCR<Npp32f>(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx.hStream);
}

NppStatus nppiColorTwist_32fC_C4IR(Npp32f * pSrcDst, int nSrcDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist_32fC_C4IR_Ctx(pSrcDst, nSrcDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}

NppStatus nppiColorTwist_32fC_C4R(const Npp32f * pSrc, int nSrcStep, Npp32f * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[4][4], const Npp32f aConstants[4]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiColorTwist_32fC_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, aTwist, aConstants, nppStreamCtx);
}
