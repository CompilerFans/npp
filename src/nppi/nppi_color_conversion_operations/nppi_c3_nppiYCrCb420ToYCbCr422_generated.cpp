// Generated API wrappers -- do not edit by hand.
// family: nppiYCrCb420ToYCbCr422


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3ToC2(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 1, 2, 0);
}

NppStatus nppiYCrCb420ToYCbCr422_8u_P3R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3Resample(pSrc, rSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx.hStream, 1, 0, 3);
}

NppStatus nppiYCrCb420ToYCbCr422_8u_P3C2R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCrCb420ToYCbCr422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCrCb420ToYCbCr422_8u_P3R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst[3], int rDstStep[3], NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCrCb420ToYCbCr422_8u_P3R_Ctx(pSrc, rSrcStep, pDst, rDstStep, oSizeROI, nppStreamCtx);
}
