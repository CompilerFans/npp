// Generated API wrappers -- do not edit by hand.
// family: nppiCFAToRGB


#include "npp.h"
#include "nppi_cfa_executor.h"

NppStatus nppiCFAToRGB_32u_C1C3R_Ctx(const Npp32u * pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcROI, Npp32u * pDst, int nDstStep, NppiBayerGridPosition eGrid, NppiInterpolationMode eInterpolation, NppStreamContext nppStreamCtx) {
  return nppi::cfa::cfaToRgb(pSrc, nSrcStep, oSrcSize, oSrcROI, pDst, nDstStep, eGrid, eInterpolation, nppStreamCtx.hStream);
}
