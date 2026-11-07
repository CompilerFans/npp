// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToYUV420


#include "npp.h"
#include "nppi_c3_twist_executor.h"

NppStatus nppiRGBToYUV420_16u_ColorTwist32f_C3P3R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::fwd420_16u(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiRGBToYUV420_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::fwd420P3_16u(pSrc, aSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, aTwist);
}
