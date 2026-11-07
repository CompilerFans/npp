// Generated API wrappers -- do not edit by hand.
// family: nppiYUV420ToRGB


#include "npp.h"
#include "nppi_c3_twist_executor.h"

NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3AC4R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], const Npp8u nAlpha, NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420(pSrc, aSrcStep, pDst, nDstStep, 4, nAlpha, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420(pSrc, aSrcStep, pDst, nDstStep, 3, 0, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3C4R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420(pSrc, aSrcStep, pDst, nDstStep, 4, 255, oSizeROI, nppStreamCtx.hStream, aTwist);
}

NppStatus nppiYUV420ToRGB_8u_ColorTwist32f_P3R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[3], int aDstStep[3], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::c3twist::inv420P3(pSrc, aSrcStep, pDst, aDstStep, oSizeROI, nppStreamCtx.hStream, aTwist);
}
