// Generated API wrappers -- do not edit by hand.
// family: nppiLUT_Trilinear


#include "npp.h"
#include "nppi_lut_executor.h"

NppStatus nppiLUT_Trilinear_8u_C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32u * pValues, Npp8u * pLevels[3], int aLevels[3], NppStreamContext nppStreamCtx) {
  return nppi::lut::trilinear(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, (const Npp8u *const *)pLevels, aLevels, true, nppStreamCtx.hStream);
}

NppStatus nppiLUT_Trilinear_8u_C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp32u * pValues, Npp8u * pLevels[3], int aLevels[3]) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiLUT_Trilinear_8u_C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pValues, pLevels, aLevels, nppStreamCtx);
}
