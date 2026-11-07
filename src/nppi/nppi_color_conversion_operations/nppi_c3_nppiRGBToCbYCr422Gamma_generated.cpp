// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToCbYCr422Gamma


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  (void)nppStreamCtx;
  return nppi::c3conv::rgbToCbYCr422GammaC3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI);
}

NppStatus nppiRGBToCbYCr422Gamma_8u_C3C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToCbYCr422Gamma_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
