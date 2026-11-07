// Generated API wrappers -- do not edit by hand.
// family: nppiBGRToCbYCr422


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiBGRToCbYCr422_8u_AC4C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::rgbToYCbCr422C3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 4, 1, 1);
}

NppStatus nppiBGRToCbYCr422_8u_AC4C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToCbYCr422_8u_AC4C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
