// Generated API wrappers -- do not edit by hand.
// family: nppiYUV420ToRGB


#include "npp.h"
#include "nppi_c3_twist_executor.h"

NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3AC4R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp16u nAlpha, NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420_16u(pSrc, aSrcStep, pDst, nDstStep, 4, nAlpha, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420_16u(pSrc, aSrcStep, pDst, nDstStep, 3, 0, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3C4R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420_16u(pSrc, aSrcStep, pDst, nDstStep, 4, 65535, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_16u_ColorTwist32f_P3R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420P3_16u(pSrc, aSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, aTwist);
}
