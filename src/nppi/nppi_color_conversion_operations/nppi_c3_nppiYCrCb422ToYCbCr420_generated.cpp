// Generated API wrappers -- do not edit by hand.
// family: nppiYCrCb422ToYCbCr420


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2ToP3(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 1, 1, 2, 0);
}

NppStatus nppiYCrCb422ToYCbCr420_8u_C2P3R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCrCb422ToYCbCr420_8u_C2P3R_Ctx(pSrc, nSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
