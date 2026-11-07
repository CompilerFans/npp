// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToCbYCr422


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiRGBToCbYCr422_8u_C3C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 3, 0, 1);
}

NppStatus nppiRGBToCbYCr422_8u_C3C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToCbYCr422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
