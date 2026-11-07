// Generated API wrappers -- do not edit by hand.
// family: nppiNV12ToRGB


#include "npp.h"
#include "nppi_nv12_color_twist_executor.h"

NppStatus nppiNV12ToRGB_16u_ColorTwist32f_P2C3R_Ctx(const Npp16u * const pSrc[2], int aSrcStep[2], Npp16u * pDst, int nDstStep, NppiSize oSizeROI, const Npp32f aTwist[3][4], NppStreamContext nppStreamCtx) {
  return nppi::nv12::nv12ToRgbColorTwist<Npp16u>(pSrc, aSrcStep, pDst, nDstStep, oSizeROI, aTwist, nppStreamCtx.hStream);
}
