// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr422ToCbYCr422


#include "npp.h"
#include "nppi_c3_conversion_executor.h"

NppStatus nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3conv::yCbCr422ToCbYCr422(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiYCbCr422ToCbYCr422_8u_C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToCbYCr422_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
