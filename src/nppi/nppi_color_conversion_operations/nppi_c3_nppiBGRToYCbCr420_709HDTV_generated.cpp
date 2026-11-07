// Generated API wrappers -- do not edit by hand.
// family: nppiBGRToYCbCr420_709HDTV


#include "npp.h"
#include "nppi_c3_color_executor.h"

NppStatus nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3color::fwd420(pSrc, nSrcStep, 4, 1, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 2, 0);
}

NppStatus nppiBGRToYCbCr420_709HDTV_8u_AC4P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiBGRToYCbCr420_709HDTV_8u_AC4P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
