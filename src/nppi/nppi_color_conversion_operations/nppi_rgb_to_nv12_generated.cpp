// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToNV12


#include "npp.h"
#include "nppi_nv12_color_twist_executor.h"

NppStatus nppiRGBToNV12_16u_ColorTwist32f_C3P2R_Ctx(const Npp16u * pSrc, int nSrcStep, Npp16u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::nv12::rgbToNv12ColorTwist<Npp16u>(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiRGBToNV12_16u_ColorTwist32f_P3P2R_Ctx(const Npp16u * const pSrc[3], int aSrcStep[3], Npp16u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::nv12::rgbToNv12ColorTwistPlanar<Npp16u>(pSrc, aSrcStep, pDst, aDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiRGBToNV12_8u_ColorTwist32f_C3P2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::nv12::rgbToNv12ColorTwist<Npp8u>(pSrc, nSrcStep, pDst, aDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}

NppStatus nppiRGBToNV12_8u_ColorTwist32f_P3P2R_Ctx(const Npp8u * const pSrc[3], int aSrcStep[3], Npp8u * pDst[2], int aDstStep[2], NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::nv12::rgbToNv12ColorTwistPlanar<Npp8u>(pSrc, aSrcStep, pDst, aDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}
