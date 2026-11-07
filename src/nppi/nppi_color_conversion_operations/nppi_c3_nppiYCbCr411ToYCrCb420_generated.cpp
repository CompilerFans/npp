// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr411ToYCrCb420


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx(const Npp8u * pSrcY, int nSrcYStep, const Npp8u * pSrcCbCr, int nSrcCbCrStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p2ToP3_420(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 1);
}

NppStatus nppiYCbCr411ToYCrCb420_8u_P2P3R(const Npp8u * pSrcY, int nSrcYStep, const Npp8u * pSrcCbCr, int nSrcCbCrStep, Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr411ToYCrCb420_8u_P2P3R_Ctx(pSrcY, nSrcYStep, pSrcCbCr, nSrcCbCrStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
