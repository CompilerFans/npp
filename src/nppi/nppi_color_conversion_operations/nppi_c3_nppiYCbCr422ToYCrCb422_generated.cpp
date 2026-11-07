// Generated API wrappers -- do not edit by hand.
// family: nppiYCbCr422ToYCrCb422


#include "npp.h"
#include "nppi_c3_yuv_layout_executor.h"

NppStatus nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::c2Reorder(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 0, 1);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3yuv::p3ToC2(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 0, 0, 1);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCrCb422_8u_C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiYCbCr422ToYCrCb422_8u_P3C2R(const Npp8u * const pSrc[3], int rSrcStep[3], Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiYCbCr422ToYCrCb422_8u_P3C2R_Ctx(pSrc, rSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
