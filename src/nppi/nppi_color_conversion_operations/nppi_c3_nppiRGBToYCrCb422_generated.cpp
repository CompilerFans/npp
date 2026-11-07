// Generated API wrappers -- do not edit by hand.
// family: nppiRGBToYCrCb422


#include "npp.h"
#include "nppi_c3_color_executor.h"

NppStatus nppiRGBToYCrCb422_8u_C3C2R_Ctx(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3color::fwdC2(pSrc, nSrcStep, 3, 0, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream, 0, 0);
}

NppStatus nppiRGBToYCrCb422_8u_P3C2R_Ctx(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx) {
  return nppi::c3color::fwdP3C2(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx.hStream);
}

NppStatus nppiRGBToYCrCb422_8u_C3C2R(const Npp8u * pSrc, int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToYCrCb422_8u_C3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}

NppStatus nppiRGBToYCrCb422_8u_P3C2R(const Npp8u * const pSrc[3], int nSrcStep, Npp8u * pDst, int nDstStep, NppiSize oSizeROI) {
  NppStreamContext nppStreamCtx;
  NppStatus status = nppGetStreamContext(&nppStreamCtx);
  if (status != NPP_NO_ERROR) {
    return status;
  }
  return nppiRGBToYCrCb422_8u_P3C2R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
}
