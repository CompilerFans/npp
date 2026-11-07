// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr420ToBGR


#include "npp.h"
#include "nppi_c3_color_executor.h"

NppStatus nppiYCbCr420ToBGR_8u_P3C4R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3],Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval, NppStreamContext nppStreamCtx) {
  return nppi::c3color::inv420(pSrc, rSrcStep, pDst, nDstStep, 4, 1, nAval, oSizeROI, nppStreamCtx.hStream, 0, 0);
}

NppStatus nppiYCbCr420ToBGR_8u_P3C4R(const Npp8u * const pSrc[3], int rSrcStep[3],Npp8u * pDst, int nDstStep, NppiSize oSizeROI, Npp8u nAval) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr420ToBGR_8u_P3C4R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nAval, nppStreamCtx);
}
