// Generated API wrappers -- do not edit by hand.
// family: nppiCbYCr422ToBGR


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiCbYCr422ToBGR_8u_C2C4R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::yCbCr422ToRgbC2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 4, 1, 1, nAval);
}

NppStatus nppiCbYCr422ToBGR_8u_C2C4R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiCbYCr422ToBGR_8u_C2C4R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nAval, nppStreamCtx);
}
