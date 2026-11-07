// Generated API wrappers -- do not edit by hand.
// family: nppiCbYCr422ToRGB


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiCbYCr422ToRGB_8u_C2C3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::yCbCr422ToRgbC2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 3, 0, 1, 0);
}

NppStatus nppiCbYCr422ToRGB_8u_C2C3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToRGB_8u_C2C3R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
